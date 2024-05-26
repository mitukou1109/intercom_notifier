import matplotlib.pyplot as plt
import numpy as np
import pydub

chime: pydub.AudioSegment = pydub.AudioSegment.from_file("chime.m4a", format="m4a")

data = np.array(chime.get_array_of_samples())
times = np.arange(data.shape[0]) / chime.frame_rate
plt.plot(times, data)
plt.show()

spectrum = np.abs(np.fft.fft(data))
frequencies = np.fft.fftfreq(spectrum.shape[0], d=1 / chime.frame_rate)
plt.plot(frequencies, spectrum)
plt.xlim(0, 2000)
plt.show()
