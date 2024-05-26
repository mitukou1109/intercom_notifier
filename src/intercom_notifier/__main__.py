import sys

from .intercom_notifier import IntercomNotifier

assert len(sys.argv) > 1

intercom_notifier = IntercomNotifier(
    stream_rate=44100,
    input_device_index=int(sys.argv[1]),
    monitor_rate=10,
    scan_duration=3.0,
    chime_frequencies=[655, 840],
    buffer_ylim=(-6000, 6000),
    spectrum_xlim=(0, 2000),
    spectrum_ylim=(0, 1e7),
)

intercom_notifier.start()

while not intercom_notifier.is_terminated():
    try:
        intercom_notifier.visualize()
    except KeyboardInterrupt:
        intercom_notifier.terminate()
