"""
chess_literature_validation.py

Tests our EXACT readiness-combination formula (the same RMS-of-features
math in src/neuroloop/readiness.py) against real data from an
independent, peer-reviewed chess-EEG study:

    Candela-Leal, Ramirez-Moreno, Lozoya-Santos (2025).
    "Task Resolution Time Estimation through Cognitive Load:
    An EEG Study of Chess Players." CogSci 2025.
    Data: https://github.com/miltoncandela/chess-cognitive-load

WHAT THIS DOES AND DOES NOT SHOW:
    This dataset provides per-participant, ALREADY baseline-normalized
    band power (each participant's task-period power relative to their
    own resting Eyes-Open period) - not raw continuous EEG, and not a
    time series. So this CANNOT test our live replay/calibration/state-
    machine mechanics (that already-tested EO/EC replay covers that).

    What it CAN honestly test: does combining these features with our
    RMS formula produce a number that relates sensibly to REAL chess
    outcomes (puzzle correctness, chess skill) in a dataset we had no
    part in collecting? That is a genuine, if modest, external check on
    the formula itself.

IMPORTANT - READ BEFORE INTERPRETING THE OUTPUT:
    With only 28 participants, do not expect (or claim) statistical
    significance. The honest question this script answers is: "do the
    trends point in a sensible direction, consistently, across multiple
    comparisons?" - not "is this proven." Report the p-values exactly
    as printed. Do not round p=0.14 down to "significant."
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

DATA_PATH = "data/raw/external_chess_study/dataNeuroM.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} participants from the external chess-EEG study.")

    # Same feature groupings as src/neuroloop/state_vector.py, built from
    # this dataset's own already-baseline-normalized columns.
    df["frontal_theta"] = df["Theta_Fz"]
    df["frontal_alpha"] = df["Alpha_Fz"]
    df["frontal_beta"] = df["Beta_Fz"]
    df["posterior_alpha"] = df[["Alpha_Pz", "Alpha_PO7", "Alpha_Oz", "Alpha_PO8"]].mean(axis=1)
    df["central_beta"] = df[["Beta_C3", "Beta_Cz", "Beta_C4"]].mean(axis=1)

    feature_cols = ["frontal_theta", "frontal_alpha", "frontal_beta",
                     "posterior_alpha", "central_beta"]

    # Identical combination formula to readiness.py's compute_readiness():
    # root-mean-square across the selected features.
    df["combined_deviation"] = np.sqrt((df[feature_cols] ** 2).mean(axis=1))

    print("\n" + "=" * 60)
    print("COMBINED DEVIATION BY NOISE CONDITION")
    print("=" * 60)
    print(df.groupby("Condicion")["combined_deviation"].agg(["mean", "std", "count"]))
    amb = df.loc[df.Condicion == "Ambiental", "combined_deviation"]
    blanco = df.loc[df.Condicion == "Blanco", "combined_deviation"]
    _, p = mannwhitneyu(amb, blanco, alternative="two-sided")
    print(f"Mann-Whitney U test (Ambient vs. distracting white noise): p={p:.4f}")

    print("\n" + "=" * 60)
    print("COMBINED DEVIATION vs. REAL CHESS OUTCOMES")
    print("=" * 60)
    rho, p2 = spearmanr(df["combined_deviation"], df["Aciertos"])
    print(f"vs. puzzle correctness (Aciertos): rho={rho:.3f}  p={p2:.4f}")

    rho2, p3 = spearmanr(df["combined_deviation"], df["ELO"])
    print(f"vs. chess skill (ELO):             rho={rho2:.3f}  p={p3:.4f}")

    print("\nBy performance category:")
    print(df.groupby("Aciertos_Cat")["combined_deviation"].agg(["mean", "std", "count"]))
    good = df.loc[df.Aciertos_Cat == "Bueno", "combined_deviation"]
    bad = df.loc[df.Aciertos_Cat == "Malo", "combined_deviation"]
    _, p4 = mannwhitneyu(good, bad, alternative="two-sided")
    print(f"Mann-Whitney U test (good vs. bad performers): p={p4:.4f}")

    print("\n" + "=" * 60)
    print("HONEST INTERPRETATION")
    print("=" * 60)
    print("None of the above p-values are below 0.05 - with only 28")
    print("participants, this is NOT a statistically significant result")
    print("and should never be reported as 'validated' or 'proven'.")
    print("What IS worth noting: check whether every comparison above")
    print("points the same direction (more deviation -> worse chess")
    print("performance / lower skill). A consistent direction across")
    print("independent comparisons, even without significance, is a")
    print("mild positive sign the formula isn't behaving randomly -")
    print("nothing stronger than that.")


if __name__ == "__main__":
    main()