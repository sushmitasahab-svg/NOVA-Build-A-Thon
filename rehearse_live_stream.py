"""
rehearse_live_stream.py

REHEARSAL / TESTING TOOL ONLY - not used during the actual live demo.

This script takes a REAL recorded EEG file and broadcasts it over LSL,
in real time, as if it were a live headset. This lets us test
LiveEEGSource's full connect -> receive -> window pipeline using real
brain-signal values, without needing the physical ANT Neuro headset
connected right now.

HOW TO USE FOR REHEARSAL:
    Terminal 1: python rehearse_live_stream.py
    Terminal 2: python run_neuroloop_demo.py --live

Terminal 1 acts as a stand-in for the real headset software. Terminal 2
is the exact same command you'd run during the actual demo - it has no
way to tell the difference between this rehearsal stream and a real one,
which is the whole point: if it works here, the only untested part left
is ANT Neuro's own software actually broadcasting correctly.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pylsl import StreamInfo, StreamOutlet
from src.preprocessing.eeg_preprocessing import load_and_preprocess

CNT_PATH = PROJECT_ROOT / "data" / "raw" / "chess_live_rehearsal" / "sasi_sasi_2026-09-12_18-57-37.cnt"


def main():
    print(f"Loading real recording for rehearsal broadcast: {CNT_PATH.name}")
    raw = load_and_preprocess(CNT_PATH)
    data = raw.get_data()  # shape (n_channels, n_samples), volts
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    n_channels = len(ch_names)

    info = StreamInfo(name="RehearsalEEG", type="EEG", channel_count=n_channels,
                       nominal_srate=sfreq, channel_format="float32",
                       source_id="rehearsal_stream_001")
    channels_xml = info.desc().append_child("channels")
    for ch in ch_names:
        c = channels_xml.append_child("channel")
        c.append_child_value("label", ch)
        c.append_child_value("unit", "volts")
        c.append_child_value("type", "EEG")

    outlet = StreamOutlet(info)
    print(f"Broadcasting '{info.name()}' - {n_channels} channels at {sfreq} Hz.")
    print("Waiting for a receiver to connect (start run_neuroloop_demo.py --live now)...")
    print("Streaming will begin immediately and run at real-time speed "
          f"for {raw.n_times / sfreq:.1f} seconds.")

    n_samples = data.shape[1]
    send_interval = 1.0 / sfreq
    start_time = time.time()

    for i in range(n_samples):
        sample = data[:, i].tolist()
        outlet.push_sample(sample)

        # Pace to real time - without this, the whole file would send
        # in under a second instead of behaving like a live headset.
        target_time = start_time + (i + 1) * send_interval
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if i % int(sfreq * 10) == 0:  # progress update every ~10s
            print(f"  ...broadcast {i/sfreq:.1f}s of {n_samples/sfreq:.1f}s")

    print("Rehearsal broadcast finished.")


if __name__ == "__main__":
    main()