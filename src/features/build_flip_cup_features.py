"""
build_flip_cup_features.py

Connects everything together for the Flip Cup dataset:

    raw .cnt files
        -> preprocessing (src/preprocessing/eeg_preprocessing.py)
        -> per-trial windows (using trigger timestamps)
        -> feature extraction (src/features/eeg_features.py)
        -> data/processed/flip_cup_features.csv

WINDOW CHOICE - explained, not arbitrary:
    We use the 5 seconds of EEG immediately BEFORE the "shot" beep for
    each trial (not after). Two reasons:

    1. Causality: the cup-flip outcome (made/missed) happens AFTER the
       beep. If we used EEG from during or after the flip, the model
       could pick up on movement artifact tied to the physical action
       itself rather than genuine pre-action brain state - that would
       not answer our real question ("does brain state predict
       performance?").

    2. Why 5 seconds specifically: we measured the actual trial_start-
       to-shot wait time across all 100 trials (both sessions) and found
       it ranges from 7.49s to 8.12s (notes.docx describes a wider
       7.5-16.5s design range, but the recorded data is tighter than
       that). A 5-second window fits safely inside every single trial's
       wait period with room to spare, and it starts well after
       trial_start (avoiding the spacebar-press/trial-start transient).
       5 seconds also gives Welch's PSD method plenty of data for a
       stable estimate.

Run this after eeg_preprocessing.py and eeg_features.py are in place.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Make sure we can import our own src/ modules regardless of where this
# script is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.eeg_preprocessing import load_and_preprocess
from src.utils.data_utils import load_trigger_file, build_trial_table
from src.features.eeg_features import extract_features

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "flip_cup"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "flip_cup_features.csv"

PRE_SHOT_WINDOW_SEC = 5.0

SESSIONS = [
    {
        "session_id": 1,
        "cnt": RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_session-01.cnt",
        "trg": RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_session-01.trg",
    },
    {
        "session_id": 2,
        "cnt": RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_session-02.cnt",
        "trg": RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_session-02.trg",
    },
]


def process_session(session_id, cnt_path, trg_path):
    """Load, preprocess, and extract features for one recording session."""
    print(f"\n--- Session {session_id} ---")
    print(f"Loading and preprocessing: {cnt_path.name}")
    raw = load_and_preprocess(cnt_path)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    print(f"Parsing trigger file: {trg_path.name}")
    trg_df = load_trigger_file(trg_path)
    trials = build_trial_table(trg_df, session_id=session_id)
    print(f"Found {len(trials)} trials.")

    rows = []
    skipped = 0
    for _, trial in trials.iterrows():
        window_end_time = trial["shot_time"]
        window_start_time = window_end_time - PRE_SHOT_WINDOW_SEC

        if window_start_time < trial["trial_start_time"]:
            # Safety check: should not happen given our measured wait
            # times, but we check rather than assume.
            skipped += 1
            continue

        start_idx = raw.time_as_index(window_start_time)[0]
        end_idx = raw.time_as_index(window_end_time)[0]

        window_data = raw.get_data(start=start_idx, stop=end_idx)  # (n_channels, n_samples)

        features = extract_features(window_data, sfreq, ch_names)

        row = {
            "participant_id": trial["participant_id"],
            "session_id": trial["session_id"],
            "trial_id": trial["trial_id"],
            "shot_time": trial["shot_time"],
            "wait_duration": trial["wait_duration"],
            "performance_label": trial["outcome_label"],
        }
        row.update(features)
        rows.append(row)

    if skipped:
        print(f"WARNING: skipped {skipped} trial(s) with a wait period "
              f"shorter than the {PRE_SHOT_WINDOW_SEC}s window.")

    return pd.DataFrame(rows)


def main():
    all_sessions = []
    for session in SESSIONS:
        df = process_session(session["session_id"], session["cnt"], session["trg"])
        all_sessions.append(df)

    features_df = pd.concat(all_sessions, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUT_PATH, index=False)

    # ---- Diagnostics ----
    id_cols = ["participant_id", "session_id", "trial_id", "shot_time",
               "wait_duration", "performance_label"]
    feature_cols = [c for c in features_df.columns if c not in id_cols]

    print("\n" + "=" * 60)
    print("FEATURE DATASET SUMMARY")
    print("=" * 60)
    print(f"Saved to: {OUT_PATH}")
    print(f"Rows (trials): {features_df.shape[0]}")
    print(f"Columns total: {features_df.shape[1]}")
    print(f"Number of EEG features: {len(feature_cols)}")
    print(f"Participants: {features_df['participant_id'].nunique()} "
          f"({features_df['participant_id'].unique().tolist()})")
    print(f"Sessions: {sorted(features_df['session_id'].unique().tolist())}")
    print("\nTarget distribution (performance_label):")
    print(features_df["performance_label"].value_counts())

    print("\nData quality checks:")
    n_nan = features_df[feature_cols].isna().sum().sum()
    n_inf = np.isinf(features_df[feature_cols].to_numpy(dtype=float)).sum()
    n_dupes = features_df.duplicated(subset=["session_id", "trial_id"]).sum()
    print(f"  NaN values in feature columns: {n_nan}")
    print(f"  Infinite values in feature columns: {n_inf}")
    print(f"  Duplicated (session_id, trial_id) rows: {n_dupes}")

    print("\nFirst 5 rows (identifier columns only, for readability):")
    print(features_df[id_cols].head())

    return features_df


if __name__ == "__main__":
    main()