import sys

from .intercom_notifier import IntercomNotifier

assert len(sys.argv) > 1

intercom_notifier = IntercomNotifier(
    stream_rate=48000,
    input_device_index=int(sys.argv[1]),
    monitor_rate=10,
    scan_duration=2.5,
)

intercom_notifier.set_visualize_param(
    buffer_ylim=(-6000, 6000),
    spectrum_xlim=(0, 2000),
    spectrum_ylim=(0, 2e6),
)

intercom_notifier.set_criteria(
    min_peak_height=0.2e6,
    min_peak_distance=500,
    chime_features=[(656.1, 0.4e6), (843.8, 0.2e6)],
    chime_frequency_tolerance=10,
)

intercom_notifier.start()

while not intercom_notifier.is_terminated():
    try:
        intercom_notifier.spin_once()
    except KeyboardInterrupt:
        intercom_notifier.terminate()
