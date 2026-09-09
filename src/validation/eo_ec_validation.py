"""
eo_ec_validation.py

Validation experiment: does our EEG pipeline detect the well-known
eyes-open vs. eyes-closed alpha effect?

WHY THIS MATTERS:
    This dataset is deliberately kept OUT of the Flip Cup ML pipeline.
    Its only job is to answer: "does our preprocessing + feature
    extraction code produce trustworthy numbers?" The eyes-open/closed
    alpha effect is one of the most well-replicated findings in EEG
    science (alpha power rises when the eyes close), so if our pipeline
    can detect it, that is real evidence our Flip Cup numbers were
    trustworthy too - it was the DATA that lacked a strong signal, not
    our code.

WHAT WE REUSE, UNCHANGED:
    - src/preprocessing/eeg_preprocessing.py (load_and_preprocess)
    - src/features/eeg_features.py (extract_features)
    Neither file needed any changes - they were written to be dataset-
    agnostic from the start.

BLOCK STRUCTURE (confirmed from brainstory_project.bst, not guessed):
    16 "Open" blocks (20.0s each) and 4 "Closed" blocks (20.0s each).
    This is an imbalanced design (4:1), which limits how much
    statistical confidence we can have in the Closed condition - we
    only have 4 independent 20-second samples of it.

A KNOWN DATA-QUALITY CAVEAT:
    When we first inspected this recording, 5 channels showed unusually
    high amplitude variance: PO7, PO4, PO6, PO5, AF8. Several of these
    sit exactly where the alpha effect is expected to be strongest
    (posterior/occipital region), so we report results for posterior
    channels split into "clean" and "flagged as noisy" groups, rather
    than silently excluding or silently trusting them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.eeg_preprocessing import load_and_preprocess
from src.features.eeg_features import extract_features
from src.validation.eo_ec_trigger_utils import load_trigger_file, build_block_table

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eyes_open_closed"
CNT_PATH = RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_EO-EC.cnt"
TRG_PATH = RAW_DIR / "Ewing_Patrick_2026-08-10_13-07-25_EO-EC.trg"

OUT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "eo_ec_features.csv"
OUT_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "eo_ec_alpha_comparison.png"
OUT_SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "eo_ec_validation_summary.csv"

# Posterior/occipital channels: where the eyes-open/closed alpha effect
# is classically strongest.
POSTERIOR_CHANNELS = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8"]
# Flagged during our earlier data-quality check (unusually high amplitude variance).
NOISY_CHANNELS = ["PO7", "PO4", "PO6", "PO5", "AF8"]


def build_feature_table():
    print(f"Loading and preprocessing: {CNT_PATH.name}")
    raw = load_and_preprocess(CNT_PATH)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    print(f"Parsing trigger file: {TRG_PATH.name}")
    trg_df = load_trigger_file(TRG_PATH)
    blocks = build_block_table(trg_df)
    print(f"Found {len(blocks)} blocks: "
          f"{blocks['condition'].value_counts().to_dict()}")

    rows = []
    for _, block in blocks.iterrows():
        start_idx = raw.time_as_index(block["start_time"])[0]
        end_idx = raw.time_as_index(block["end_time"])[0]
        window_data = raw.get_data(start=start_idx, stop=end_idx)

        features = extract_features(window_data, sfreq, ch_names)
        row = {
            "block_id": block["block_id"],
            "condition": block["condition"],
            "start_time": block["start_time"],
            "duration": block["duration"],
        }
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


def channel_alpha_relative_column(ch):
    return f"{ch}_alpha_relative"


def compare_condition(df, channels, group_label):
    """
    For a group of channels, average their alpha_relative power per
    block, then compare Open vs Closed with a Mann-Whitney U test
    (chosen over a t-test because we only have 4 Closed blocks - too
    few to safely assume a normal distribution).
    """
    cols = [channel_alpha_relative_column(ch) for ch in channels
            if channel_alpha_relative_column(ch) in df.columns]
    missing = [ch for ch in channels if channel_alpha_relative_column(ch) not in df.columns]
    if missing:
        print(f"  NOTE: channels not found in feature table, skipped: {missing}")

    df = df.copy()
    df["_avg_alpha"] = df[cols].mean(axis=1)

    open_vals = df.loc[df["condition"] == "Open", "_avg_alpha"]
    closed_vals = df.loc[df["condition"] == "Closed", "_avg_alpha"]

    stat, p_value = mannwhitneyu(open_vals, closed_vals, alternative="two-sided")

    result = {
        "group": group_label,
        "n_channels": len(cols),
        "n_open_blocks": len(open_vals),
        "n_closed_blocks": len(closed_vals),
        "open_mean_alpha_relative": float(open_vals.mean()),
        "closed_mean_alpha_relative": float(closed_vals.mean()),
        "difference_closed_minus_open": float(closed_vals.mean() - open_vals.mean()),
        "mannwhitney_u": float(stat),
        "p_value": float(p_value),
    }
    return result, open_vals, closed_vals


def main():
    df = build_feature_table()

    OUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FEATURES_PATH, index=False)
    print(f"\nSaved block-level feature table to: {OUT_FEATURES_PATH}")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

    all_channels = [c.split("_alpha_relative")[0] for c in df.columns
                    if c.endswith("_alpha_relative")]
    clean_posterior = [ch for ch in POSTERIOR_CHANNELS if ch not in NOISY_CHANNELS]
    noisy_posterior = [ch for ch in POSTERIOR_CHANNELS if ch in NOISY_CHANNELS]

    print("\n" + "=" * 60)
    print("ALPHA POWER: EYES OPEN vs EYES CLOSED")
    print("=" * 60)
    print("(Expected direction, based on established EEG science: "
          "Closed > Open)\n")

    groups = [
        ("all_channels", all_channels),
        ("posterior_clean", clean_posterior),
        ("posterior_flagged_noisy", noisy_posterior),
    ]

    summary_rows = []
    plot_data = {}
    for label, channels in groups:
        print(f"--- {label} ({len(channels)} channels: {channels}) ---")
        result, open_vals, closed_vals = compare_condition(df, channels, label)
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.5f}")
            else:
                print(f"  {k}: {v}")
        direction = "Closed > Open (matches expectation)" if result["difference_closed_minus_open"] > 0 \
            else "Closed < Open (opposite of expectation)"
        significant = "YES (p < 0.05)" if result["p_value"] < 0.05 else "NO (p >= 0.05)"
        print(f"  Direction: {direction}")
        print(f"  Statistically significant: {significant}\n")

        summary_rows.append(result)
        plot_data[label] = (open_vals, closed_vals)

    summary_df = pd.DataFrame(summary_rows)
    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY_PATH, index=False)
    print(f"Saved summary to: {OUT_SUMMARY_PATH}")

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (label, _) in zip(axes, groups):
        open_vals, closed_vals = plot_data[label]
        ax.boxplot([open_vals, closed_vals], tick_labels=["Open", "Closed"])
        ax.set_title(label)
        ax.set_ylabel("Average relative alpha power")
    plt.suptitle("Eyes Open vs Eyes Closed: Relative Alpha Power by Channel Group")
    plt.tight_layout()
    OUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIGURE_PATH, dpi=120)
    print(f"Saved figure to: {OUT_FIGURE_PATH}")

    print("\n" + "=" * 60)
    print("HONEST CAVEATS")
    print("=" * 60)
    print("- Only 4 Closed blocks exist in this recording (vs 16 Open) - ")
    print("  the Closed-condition estimate rests on very few independent samples.")
    print("- 'posterior_flagged_noisy' channels showed unusually high amplitude")
    print("  variance during our earlier data-quality check - treat that group's")
    print("  result with extra caution, it is reported for transparency, not as")
    print("  equally trustworthy evidence.")


if __name__ == "__main__":
    main()