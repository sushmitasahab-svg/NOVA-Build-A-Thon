"""
readiness.py

Turns a dict of baseline deviations (z-scores, from baseline.py) into
one number: a 0-100 READINESS SCORE (Section 12 of the spec).

WHAT THIS SCORE MEANS - AND DOES NOT MEAN:
    "How stable is the participant's current EEG-derived state relative
    to their own calibrated baseline?" A score near 100 means the
    current window looks like their calibration period. A low score
    means it looks quite different.

    It does NOT mean "how likely to make a good chess move," "how
    focused," or anything about chess performance specifically. It is a
    deliberately narrow, honest measurement of EEG self-similarity.

HOW IT'S COMPUTED:
    1. Combine every feature's z-score into one overall "distance from
       baseline" number, using the root-mean-square (RMS) of the
       z-scores. RMS is used (rather than, say, the max) so that no
       single noisy feature can dominate the score by itself, while
       still being sensitive to a genuinely large, broad deviation.
    2. Map that distance to 0-100 with a smooth exponential decay, so
       the score is always bounded, and larger deviations always
       produce a lower (never negative) score.
"""

import numpy as np

from src.neuroloop import config


def compute_readiness(deviations, quality):
    """
    Compute one window's raw (not yet temporally smoothed) readiness
    result.

    Parameters
    ----------
    deviations : dict
        feature_name -> z-score, as returned by PersonalBaseline.deviation()
    quality : float
        Signal quality score for this window, 0.0-1.0 (see state_vector.py)

    Returns
    -------
    dict with:
        raw_score : float (0-100), the unsmoothed readiness estimate
        combined_deviation : float, the RMS z-score behind that estimate
        top_contributors : list of (feature_name, z_score), the 3
            features that deviated most from baseline this window -
            for the "why did this change" explanation, never presented
            as proof of a specific mental state (Section 22).
        reliable : bool, False if quality is too low to trust this
            window's estimate at all (Section 6)
    """
    reliable = quality >= config.QUALITY_THRESHOLD

    if not deviations:
        combined_deviation = 0.0
    else:
        z_values = np.array(list(deviations.values()))
        combined_deviation = float(np.sqrt(np.mean(z_values ** 2)))

    raw_score = 100.0 * np.exp(-config.READINESS_DECAY_RATE * combined_deviation)

    top_contributors = sorted(deviations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    return {
        "raw_score": float(raw_score),
        "combined_deviation": combined_deviation,
        "top_contributors": top_contributors,
        "reliable": reliable,
        "quality": quality,
    }