import math
import threading
import time

import matplotlib.axes
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import scipy.signal


class IntercomNotifier:

    def __init__(
        self,
        *,
        stream_rate: int,
        input_device_index: int,
        monitor_rate: int,
        scan_duration: float,
        chime_frequencies: list[int],
        buffer_ylim: tuple[float, float],
        spectrum_xlim: tuple[float, float],
        spectrum_ylim: tuple[float, float],
    ) -> None:
        self.stream_rate = stream_rate
        self.stream_buffer_size = stream_rate // monitor_rate
        self.scan_duration = scan_duration
        self.chime_frequencies = chime_frequencies

        self.lock = threading.Lock()

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=stream_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=self.stream_buffer_size,
        )

        self.buffer: np.ndarray = np.zeros(
            1 << math.ceil(scan_duration * stream_rate).bit_length(), dtype=np.int16
        )
        self.frequencies: np.ndarray = np.array([])
        self.spectrum: np.ndarray = np.array([])
        self.peaks: np.ndarray = np.array([])

        self.buffer_ax: matplotlib.axes.Axes
        self.spectrum_ax: matplotlib.axes.Axes
        self.fig, (self.buffer_ax, self.spectrum_ax) = plt.subplots(1, 2)

        (self.buffer_line,) = self.buffer_ax.plot(
            np.linspace(-scan_duration, 0, self.buffer.shape[0]), self.buffer
        )
        self.buffer_ax.set_ylim(buffer_ylim)

        (self.spectrum_line,) = self.spectrum_ax.plot([])
        (self.peak_line,) = self.spectrum_ax.plot([], [], "x")
        self.spectrum_ax.set_xlim(spectrum_xlim)
        self.spectrum_ax.set_ylim(spectrum_ylim)

        def on_key_press(event: matplotlib.backend_bases.KeyEvent) -> None:
            if event.key == "q":
                self.terminate()

        self.fig.canvas.mpl_connect("key_press_event", on_key_press)

        self.terminated = False
        self.chime_detected = False
        self.last_detected_time = 0

    def start(self) -> None:
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()

    def terminate(self) -> None:
        self.terminated = True
        self.monitor_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def is_terminated(self) -> bool:
        return self.terminated

    def monitor(self) -> None:
        while not self.terminated:
            data = np.frombuffer(
                self.stream.read(self.stream_buffer_size), dtype=np.int16
            )
            buffer = np.roll(self.buffer, -data.shape[0])
            buffer[-data.shape[0] :] = data

            spectrum = np.abs(np.fft.fft(buffer))
            frequencies = np.fft.fftfreq(spectrum.shape[0], d=1 / self.stream_rate)

            peaks: np.ndarray
            peaks, _ = scipy.signal.find_peaks(spectrum, height=2e5, distance=800)

            with self.lock:
                self.buffer = buffer
                self.frequencies = frequencies
                self.spectrum = spectrum
                self.peaks = peaks

            if peaks.size > 0:
                has_all_chime_frequencies = all(
                    np.any(np.isclose(frequencies[peaks], frequency, atol=50))
                    for frequency in self.chime_frequencies
                )

                if has_all_chime_frequencies:
                    self.chime_detected = True
                    if time.time() - self.last_detected_time > self.scan_duration:
                        print("Chime detected")
                        self.last_detected_time = time.time()

    def visualize(self) -> None:
        with self.lock:
            buffer = self.buffer
            frequencies = self.frequencies
            spectrum = self.spectrum
            peaks = self.peaks

        self.buffer_line.set_ydata(buffer)
        self.spectrum_line.set_data(frequencies, spectrum)

        if peaks.size > 0:
            self.peak_line.set_data(frequencies[peaks], spectrum[peaks])

        try:
            plt.pause(0.01)
        except ValueError:
            pass
