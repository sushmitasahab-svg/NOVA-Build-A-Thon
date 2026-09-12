"""
state_vector.py

Turns one EEG window into a small set of NAMED, interpretable raw
features - the ingredients that will later be compared against the
participant's personal baseline (see baseline.py) to produce the
STATE VECTOR described in Section 10 of the spec.

We deliberately compute a SMALL, curated set of features here rather
than reusing eeg_features.py's full extract_features() (which produces
~15 features per channel). That function is still used internally
(band_powers_for_channel, time_domain_features) - we're not
duplicating logic, just being selective about what goes into the
cognitive-state model, so the eventual "why did the state change"
explanation stays understandable rather than a wall of numbers.

WHAT EACH FEATURE MEANS (and does NOT mean):
    frontal_theta / frontal_alpha / frontal_beta:
        Relative band power at Fz (our only frontal electrode). These
        are commonly studied in cognitive-load research, but they are
        MODEL INPUTS, not proof of any specific mental state on their
        own (per Section 2's requirement).
    frontal_theta_beta_ratio:
        theta/beta power ratio at Fz - a commonly used ratio in the
        literature, again treated only as a feature, not a verdict.
    posterior_alpha:
        Average relative alpha power across our posterior channels
        (Pz, PO7, Oz, PO8) - the same kind of signal our Eyes-Open/
        Closed validation already proved this pipeline can measure
        reliably.
    central_beta:
        Average relative beta power across C3/Cz/C4 - carried over
        from our Flip Cup literature-guided feature work.
    global_power:
        Average total band power (delta+theta+alpha+beta) across all
        available channels - a general "how much EEG activity" measure.
    eeg_variability:
        Average standard deviation of the raw (filtered) signal across
        channels - a simple measure of how much the signal is moving
        around, independent of frequency content.
"""

import numpy as np

from src.features.eeg_features import band_powers_for_channel, time_domain_features
from src.neuroloop import config


def compute_signal_quality(window_data):
    """
    Fraction of channels (0.0-1.0) whose standard deviation falls in a
    plausible EEG range. Too low = likely flat/disconnected. Too high =
    likely saturated or dominated by a large artifact.
    """
    stds = window_data.std(axis=1)
    usable = [(config.QUALITY_MIN_STD_VOLTS < s < config.QUALITY_MAX_STD_VOLTS) for s in stds]
    if not usable:
        return 0.0
    return sum(usable) / len(usable)


def compute_raw_state_features(window_data, sfreq, ch_names):
    """
    Compute the curated raw feature set for one EEG window.

    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
    ch_names : list of str, same order as window_data's rows

    Returns
    -------
    dict mapping feature name -> value (not yet baseline-normalized)
    """
    per_channel_relative = {}
    per_channel_total_power = {}
    for i, ch in enumerate(ch_names):
        absolute, relative = band_powers_for_channel(window_data[i, :], sfreq)
        per_channel_relative[ch] = relative
        per_channel_total_power[ch] = sum(absolute.values())

    features = {}

    if config.FRONTAL_CHANNEL in per_channel_relative:
        rel = per_channel_relative[config.FRONTAL_CHANNEL]
        features["frontal_theta"] = rel["theta"]
        features["frontal_alpha"] = rel["alpha"]
        features["frontal_beta"] = rel["beta"]
        features["frontal_theta_beta_ratio"] = (
            rel["theta"] / rel["beta"] if rel["beta"] > 1e-12 else 0.0
        )

    posterior_present = [ch for ch in config.POSTERIOR_CHANNELS if ch in per_channel_relative]
    if posterior_present:
        features["posterior_alpha"] = float(np.mean(
            [per_channel_relative[ch]["alpha"] for ch in posterior_present]
        ))

    central_present = [ch for ch in config.CENTRAL_CHANNELS if ch in per_channel_relative]
    if central_present:
        features["central_beta"] = float(np.mean(
            [per_channel_relative[ch]["beta"] for ch in central_present]
        ))

    if per_channel_total_power:
        features["global_power"] = float(np.mean(list(per_channel_total_power.values())))

    stds = [time_domain_features(window_data[i, :])["std"] for i in range(len(ch_names))]
    features["eeg_variability"] = float(np.mean(stds))

    return features