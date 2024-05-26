import sys

from .intercom_notifier import IntercomNotifier

assert len(sys.argv) > 1

intercom_notifier = IntercomNotifier(
    stream_rate=44100,
    input_device_index=int(sys.argv[1]),
    monitor_rate=10,
    scan_duration=2.5,
)

intercom_notifier.set_visualize_param(
    buffer_ylim=(-6000, 6000),
    spectrum_xlim=(0, 2000),
    spectrum_ylim=(0, 4e7),
)

intercom_notifier.set_criteria(
    min_peak_height=0.5e7,
    min_peak_distance=100,
    chime_features=[(656.1, 2e7), (843.8, 0.5e7)],
    chime_frequency_tolerance=10,
)

intercom_notifier.start()

while not intercom_notifier.is_terminated():
    try:
        intercom_notifier.visualize()
    except KeyboardInterrupt:
        intercom_notifier.terminate()
