"""
run_neuroloop_demo.py

The script you actually run. Loads an EEG recording, replays it as if
it were streaming live, and prints real-time-style terminal feedback
using the full NeuroLoop chain:

    ReplayEEGSource -> state_vector -> PersonalBaseline
        -> readiness -> smoothing -> state_machine -> feedback_cli

USAGE:
    python run_neuroloop_demo.py
    python run_neuroloop_demo.py --source eo_ec --max-time 200
    python run_neuroloop_demo.py --source chess_v2

Add new recordings to the RECORDINGS dict below as they become
available.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.eeg_preprocessing import load_and_preprocess
from src.preprocessing.unicorn_preprocessing import load_and_preprocess_unicorn_csv
from src.neuroloop.eeg_source import ReplayEEGSource
from src.neuroloop.state_vector import compute_raw_state_features, compute_signal_quality
from src.neuroloop.baseline import PersonalBaseline
from src.neuroloop.readiness import compute_readiness
from src.neuroloop.smoothing import ReadinessSmoother
from src.neuroloop.state_machine import ReadinessStateMachine
from src.neuroloop import feedback_cli, config
from src.neuroloop.music_controller import MusicController

# Each entry: path to the recording, which loader function to use, the
# calibration duration for that session, and (optionally) real move
# timestamps for reference annotations printed alongside the feedback.
RECORDINGS = {
    "eo_ec": {
        "path": PROJECT_ROOT / "data" / "raw" / "eyes_open_closed" /
                "Ewing_Patrick_2026-08-10_13-07-25_EO-EC.cnt",
        "loader": load_and_preprocess,
        "calibration": config.CALIBRATION_DURATION_SEC,
        "move_timestamps": None,
    },
    "chess_v2": {
        "path": PROJECT_ROOT / "data" / "raw" / "chess_unicorn_v2" /
                "UnicornRecorder_12_09_2026_15_41_290.csv",
        "loader": load_and_preprocess_unicorn_csv,
        # Real resting/calibration period for this session, from the
        # manually-timed laps (first lap = 2:31.13 = 151.13s).
        "calibration": 151.13,
        # Cumulative move timestamps (seconds), from the same manual
        # lap timings - printed as reference markers only, not used in
        # any calculation.
        "move_timestamps": [
            151.13, 156.53, 164.17, 174.62, 186.38, 201.17, 207.97, 215.27,
            232.89, 248.21, 261.38, 275.20, 280.92, 288.84, 305.61, 312.91,
            335.86, 348.56, 361.01, 374.26, 382.26, 407.38, 426.51, 438.57,
            446.19, 463.70, 475.92, 487.91, 492.65, 508.03, 514.64, 530.84,
            542.06, 550.06, 569.29, 572.35,
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroLoop terminal feedback demo")
    parser.add_argument("--source", choices=list(RECORDINGS.keys()), default="eo_ec",
                         help="Which recording to replay (see RECORDINGS dict).")
    parser.add_argument("--calibration", type=float, default=None,
                         help="Override the calibration period length in seconds "
                              "(defaults to the value set for the chosen --source).")
    parser.add_argument("--max-time", type=float, default=None,
                         help="Stop replay after this many seconds of recording time "
                              "(useful for quick tests instead of running the whole file).")
    return parser.parse_args()


def main():
    args = parse_args()
    entry = RECORDINGS[args.source]
    cnt_path = entry["path"]
    loader = entry["loader"]
    calibration_duration = args.calibration if args.calibration is not None else entry["calibration"]
    move_timestamps = entry.get("move_timestamps")

    print(f"Loading and preprocessing: {cnt_path.name}")
    raw = loader(cnt_path)
    source = ReplayEEGSource(raw)

    missing = [c for c in config.CHANNELS_OF_INTEREST if c not in source.ch_names]
    if missing:
        print(f"ERROR: this recording is missing required channels: {missing}")
        print("Cannot proceed - the state vector needs all of "
              f"{config.CHANNELS_OF_INTEREST}.")
        return
    channel_idx = [source.ch_names.index(c) for c in config.CHANNELS_OF_INTEREST]

    print(feedback_cli.format_calibration_start(calibration_duration))

    baseline = PersonalBaseline()
    smoother = ReadinessSmoother()
    state_machine = ReadinessStateMachine()
    calibrated = False
    next_move_idx = 0

    has_more = True
    while has_more:
        window = source.get_window(config.WINDOW_LENGTH_SEC)

        # Print a reference marker when we cross a real, manually-timed
        # chess move. This is annotation only - it does not feed into
        # any calculation, it's purely so we can visually compare the
        # feedback loop's behavior against when moves actually happened.
        if move_timestamps:
            while (next_move_idx < len(move_timestamps)
                   and source.current_time >= move_timestamps[next_move_idx]):
                print(f"           .... MOVE {next_move_idx + 1} made "
                      f"around t={move_timestamps[next_move_idx]:.1f}s ....")
                next_move_idx += 1

        if window is not None:
            window_subset = window[channel_idx, :]
            quality = compute_signal_quality(window_subset)
            raw_features = compute_raw_state_features(
                window_subset, source.sfreq, config.CHANNELS_OF_INTEREST
            )

            if source.current_time <= calibration_duration:
                baseline.add_calibration_sample(raw_features)

            else:
                if not calibrated:
                    baseline.fit()
                    calibrated = True
                    print(feedback_cli.format_baseline_established(
                        source.current_time, baseline.summary()
                    ))

                if quality < config.QUALITY_THRESHOLD:
                    print(feedback_cli.format_signal_uncertain(source.current_time, quality))
                    smoother.update(0.0, reliable=False)
                    state_machine.update(smoother.value, source.current_time)

                else:
                    deviations = baseline.deviation(raw_features)
                    readiness_result = compute_readiness(deviations, quality)
                    smoothed = smoother.update(
                        readiness_result["raw_score"], readiness_result["reliable"]
                    )
                    transition = state_machine.update(smoothed, source.current_time)

                    print(feedback_cli.format_status_line(
                        source.current_time, transition["state"], smoothed, quality
                    ))
                    if transition["changed"]:
                        print(feedback_cli.format_transition_block(
                            source.current_time, transition, readiness_result, smoothed
                        ))

        has_more = source.advance(config.STEP_SEC)
        if args.max_time is not None and source.current_time > args.max_time:
            break

    print(feedback_cli.format_final_summary(source.current_time, state_machine.state))


if __name__ == "__main__":
    main()