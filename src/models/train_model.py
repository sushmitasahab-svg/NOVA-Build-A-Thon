"""
train_model.py

Stage 2-5: baseline model, Logistic Regression, Random Forest, and
session-aware cross-validation for the Flip Cup made/missed prediction
task.

WHY SESSION-LEVEL GROUPED CROSS-VALIDATION:
    We have 100 trials from ONE participant across 2 sessions (50 each).
    There is no way to hold out a different person, so the least-leaky
    validation we can honestly do is: train on one session, test on the
    other. This is implemented with GroupKFold(n_splits=2), grouping by
    session_id. This means there are only 2 folds - not because we chose
    a small number, but because we only have 2 valid groups.

WHY FEATURE SELECTION FOR LOGISTIC REGRESSION:
    We have 930 features and only ~50 training samples per fold. Feeding
    all 930 into Logistic Regression would badly overfit. We use
    SelectKBest (ANOVA F-test) to keep the 20 most individually
    informative features, and this is done INSIDE the pipeline, so it is
    refit only on the training fold each time - the test fold never
    influences which features get selected. k=20 is a conservative,
    round-number starting point (far fewer features than training
    samples), not a value we searched for and picked because it looked
    best.

WHY RANDOM FOREST HYPERPARAMETERS:
    max_depth=5 and min_samples_leaf=3 deliberately keep trees small.
    With only ~50 training samples, deep unrestricted trees would
    memorize the training data instead of learning general patterns.

WHAT THIS SCRIPT DOES NOT DO:
    It does not search for the "best" hyperparameters by trying many
    options and picking whichever scored highest on the test fold - that
    itself would be a form of leakage. These are fixed, reasoned
    choices, not tuned choices.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "flip_cup_features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

ID_COLS = ["participant_id", "session_id", "trial_id", "shot_time",
           "wait_duration", "performance_label"]

# zero_division=0 makes precision/recall/f1 return 0.0 (instead of warning
# and guessing) when a fold predicts only one class - which happens with the
# majority-class baseline, so this keeps the output clean and well-defined.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
}
SCORING_NAMES = list(SCORING.keys())

RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    X = df[feature_cols].to_numpy(dtype=float)
    # made=1, missed=0. We fix this mapping explicitly rather than trusting
    # alphabetical LabelEncoder ordering, so the meaning of "class 1" is
    # never ambiguous.
    y = (df["performance_label"] == "made").astype(int).to_numpy()
    groups = df["session_id"].to_numpy()

    return df, X, y, groups, feature_cols


def summarize_cv(name, cv_results, n_splits):
    """Print per-fold and mean+-std results for one model."""
    print(f"\n--- {name} ---")
    for metric in SCORING:
        key = f"test_{metric}"
        scores = cv_results[key]
        fold_str = ", ".join(f"fold{i+1}={s:.3f}" for i, s in enumerate(scores))
        print(f"  {metric:18s}: mean={scores.mean():.3f}  std={scores.std():.3f}  ({fold_str})")


def cv_results_to_dict(name, cv_results):
    out = {"model": name}
    for metric in SCORING:
        key = f"test_{metric}"
        scores = cv_results[key]
        out[f"{metric}_mean"] = float(scores.mean())
        out[f"{metric}_std"] = float(scores.std())
        for i, s in enumerate(scores):
            out[f"{metric}_fold{i+1}"] = float(s)
    return out


def main():
    df, X, y, groups, feature_cols = load_dataset()

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Samples: {X.shape[0]}   Features: {X.shape[1]}")
    print(f"Target distribution: made={y.sum()}  missed={(y == 0).sum()}")
    print(f"Groups (sessions): {sorted(set(groups.tolist()))}")

    n_splits = len(set(groups.tolist()))
    gkf = GroupKFold(n_splits=n_splits)

    # ---- Baseline: always predict the majority class ----
    baseline = DummyClassifier(strategy="most_frequent")
    baseline_cv = cross_validate(baseline, X, y, groups=groups, cv=gkf,
                                  scoring=SCORING)
    summarize_cv("Baseline (majority class)", baseline_cv, n_splits)

    # ---- Model 1: Logistic Regression with feature selection, in a Pipeline ----
    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=20)),
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                    random_state=RANDOM_STATE)),
    ])
    logreg_cv = cross_validate(logreg_pipeline, X, y, groups=groups, cv=gkf,
                                scoring=SCORING)
    summarize_cv("Logistic Regression", logreg_cv, n_splits)

    # ---- Model 2: Random Forest, no scaling needed ----
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                        min_samples_leaf=3,
                                        random_state=RANDOM_STATE)),
    ])
    rf_cv = cross_validate(rf_pipeline, X, y, groups=groups, cv=gkf,
                            scoring=SCORING)
    summarize_cv("Random Forest", rf_cv, n_splits)

    # ---- Comparison table ----
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (mean across folds)")
    print("=" * 60)
    comparison_rows = [
        cv_results_to_dict("baseline_majority_class", baseline_cv),
        cv_results_to_dict("logistic_regression", logreg_cv),
        cv_results_to_dict("random_forest", rf_cv),
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    display_cols = ["model"] + [f"{m}_mean" for m in SCORING]
    print(comparison_df[display_cols].to_string(index=False))

    # ---- Save metrics ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved metrics to: {METRICS_DIR / 'model_comparison.csv'}")

    # ---- Save fitted pipelines (fit on ALL 100 trials, for later use/inspection) ----
    # NOTE: these "final fit" models are for saving/inspection and later use in
    # predict.py. The performance numbers we trust are the cross-validation
    # scores above, NOT this final fit's training accuracy - fitting on all
    # data does not tell us how well it generalizes.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_pipeline.fit(X, y)
    rf_pipeline.fit(X, y)
    joblib.dump(logreg_pipeline, MODELS_DIR / "logistic_regression_pipeline.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest_pipeline.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    print(f"Saved fitted pipelines to: {MODELS_DIR}")

    # ---- Honest interpretation flags ----
    print("\n" + "=" * 60)
    print("INTERPRETATION CHECKS")
    print("=" * 60)
    baseline_bal_acc = baseline_cv["test_balanced_accuracy"].mean()
    logreg_bal_acc = logreg_cv["test_balanced_accuracy"].mean()
    rf_bal_acc = rf_cv["test_balanced_accuracy"].mean()

    for name, bal_acc in [("Logistic Regression", logreg_bal_acc),
                           ("Random Forest", rf_bal_acc)]:
        diff = bal_acc - baseline_bal_acc
        if diff > 0.05:
            verdict = f"performs {diff:.3f} better than baseline (balanced accuracy) - some signal detected."
        elif diff < -0.05:
            verdict = f"performs {abs(diff):.3f} WORSE than baseline (balanced accuracy)."
        else:
            verdict = "is within +/-0.05 of baseline (balanced accuracy) - no clear evidence of predictive signal."
        print(f"{name}: {verdict}")

    print("\nRemember: with only 2 session-based folds, these numbers are based on")
    print("just 2 train/test splits (50 vs 50 trials each). Treat any single")
    print("fold's score with caution - look at both fold values above, not just the mean.")


if __name__ == "__main__":
    main()