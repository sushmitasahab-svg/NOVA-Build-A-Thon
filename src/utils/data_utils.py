"""
data_utils.py

Small helper functions for reading ANT Neuro trigger (.trg) files and
turning the raw list of trigger codes into a clean, one-row-per-trial
table for the Flip Cup experiment.

Why this file exists:
Our EEG recordings (.cnt files) come with a trigger channel that marks
important moments in time (trial start, beep, made/missed shot).
Before we can cut the continuous EEG into per-trial windows, we need to
turn that trigger list into a simple table: one row per trial, with the
timestamps we care about and the outcome label.
"""

import pandas as pd

# From data/raw/flip_cup/Marker definitions.xlsx (ANT Neuro / Johnathan Drucker)
TRIAL_START_CODE = 1
SHOT_CODE = 2
MADE_SHOT_CODE = 4
MISSED_SHOT_CODE = 8
EXP_START_CODE = 64
EXP_STOP_CODE = 128


def load_trigger_file(trg_path):
    """
    Read a .trg file into a DataFrame with columns: time_s, sample, code.

    ANT Neuro .trg files are plain text, one trigger per line:
        <time_in_seconds> <sample_index> <code>

    Some lines (the very first header-like line, or a "__" marker) do not
    have a numeric code. We drop those here since they are not one of our
    experiment's trial-relevant markers.
    """
    df = pd.read_csv(trg_path, sep=r"\s+", header=None,
                      names=["time_s", "sample", "code"])
    df = df.dropna(subset=["code"])
    # Keep only rows where the code is actually a number (drops "__" etc.)
    df = df[pd.to_numeric(df["code"], errors="coerce").notna()].copy()
    df["code"] = df["code"].astype(int)
    df = df.reset_index(drop=True)
    return df


def build_trial_table(trg_df, session_id, participant_id="patrick_ewing"):
    """
    Turn the raw trigger table into one row per trial.

    Each trial in this experiment looks like:
        code 1 (trial_start) -> variable wait -> code 2 (shot/beep)
        -> the person flips the cup -> code 4 (made) or code 8 (missed)

    We pair up trial_start events with the next shot event, and the next
    made/missed event, in the order they occur. This assumes trials do not
    overlap and appear in order, which is true for this experiment (one
    trial fully finishes before the next begins).

    Returns a DataFrame with:
        participant_id, session_id, trial_id,
        trial_start_time, shot_time, outcome_time,
        wait_duration, outcome_code, outcome_label
    """
    starts = trg_df.loc[trg_df["code"] == TRIAL_START_CODE, "time_s"].reset_index(drop=True)
    shots = trg_df.loc[trg_df["code"] == SHOT_CODE, "time_s"].reset_index(drop=True)
    outcomes = trg_df.loc[trg_df["code"].isin([MADE_SHOT_CODE, MISSED_SHOT_CODE]),
                           ["time_s", "code"]].reset_index(drop=True)

    n_trials = min(len(starts), len(shots), len(outcomes))
    if not (len(starts) == len(shots) == len(outcomes)):
        print(f"[data_utils] WARNING: mismatched counts in session {session_id} - "
              f"trial_start={len(starts)}, shot={len(shots)}, outcome={len(outcomes)}. "
              f"Using the first {n_trials} of each (matched in order).")

    rows = []
    for i in range(n_trials):
        trial_start_time = starts[i]
        shot_time = shots[i]
        outcome_time = outcomes.loc[i, "time_s"]
        outcome_code = outcomes.loc[i, "code"]
        outcome_label = "made" if outcome_code == MADE_SHOT_CODE else "missed"

        rows.append({
            "participant_id": participant_id,
            "session_id": session_id,
            "trial_id": i,
            "trial_start_time": trial_start_time,
            "shot_time": shot_time,
            "outcome_time": outcome_time,
            "wait_duration": shot_time - trial_start_time,
            "outcome_code": outcome_code,
            "outcome_label": outcome_label,
        })

    return pd.DataFrame(rows)