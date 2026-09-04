"""
eeg_features.py

Reusable functions that turn a short EEG window into a set of numbers
(features) that a machine learning model can use.

Background, in plain language:
    Raw EEG is just a long list of voltage measurements over time, one
    list per electrode. A machine learning model can't make much sense
    of millions of raw numbers directly. Instead, we summarize each
    short window of EEG using a handful of meaningful numbers - this is
    "feature extraction".

    The most common and interpretable EEG features are FREQUENCY-BAND
    POWERS. Brain activity naturally contains rhythms at different
    speeds (frequencies), and these rhythms are grouped into named
    bands:

        delta  (0.5-4 Hz)  - slowest rhythm, dominant in deep sleep
        theta  (4-8 Hz)    - drowsiness, some memory/attention tasks
        alpha  (8-13 Hz)   - relaxed wakefulness, strongly linked to
                             eyes-open vs eyes-closed
        beta   (13-30 Hz)  - active thinking, focus, motor planning
        gamma  (30+ Hz)    - higher-level processing (we skip this for
                             now - see note below)

    "Power" in a given band means: how much of the signal's energy is
    concentrated at those frequencies. We estimate this using Welch's
    Power Spectral Density (PSD) method.

What is Welch's PSD, and why use it here?
    A raw EEG segment doesn't tell you "how much alpha is in this
    signal" just by looking at the numbers - you first have to convert
    the signal from the TIME domain (voltage vs. time) into the
    FREQUENCY domain (power vs. frequency). Welch's method does this by
    splitting the window into several overlapping smaller chunks,
    computing a Fourier transform on each chunk, and averaging the
    results. Averaging makes the estimate much less noisy than a single
    Fourier transform on the whole window, which is why Welch's method
    is the standard choice for EEG band-power features.

Why we skip gamma for now:
    Reliable gamma power estimation typically wants a higher sampling
    rate and very clean data (gamma easily gets contaminated by muscle
    artifact). Our recordings are sampled at 512 Hz, which technically
    allows it, but we're deliberately starting with the four
    well-established, lower-risk bands first, per the "start with a
    small set of interpretable features" instruction. We can add gamma
    later if the simpler features look promising.
"""

import numpy as np
from scipy.signal import welch

# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _welch_psd(signal_1d, sfreq):
    """
    Compute the Power Spectral Density of a single-channel signal using
    Welch's method.

    nperseg (samples per segment) is capped at the window length so this
    also works safely on short windows. Using 1-second segments (sfreq
    samples) gives about 1 Hz of frequency resolution, which is enough
    to tell our bands apart.
    """
    nperseg = min(len(signal_1d), int(sfreq))
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=nperseg)
    return freqs, psd


def band_power(signal_1d, sfreq, band):
    """
    Average power of signal_1d within a given (low_hz, high_hz) band.

    We integrate (sum, weighted by frequency spacing) the PSD across the
    frequencies that fall inside the band.
    """
    freqs, psd = _welch_psd(signal_1d, sfreq)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    # np.trapz was renamed to np.trapezoid in newer NumPy versions; support both.
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(psd[mask], freqs[mask]))


def band_powers_for_channel(signal_1d, sfreq):
    """
    Compute absolute and relative power in each band (delta/theta/alpha/beta)
    for one channel's signal.

    Absolute power: raw power in that band (units depend on the signal).
    Relative power: that band's power divided by the total power across
    all four bands. Relative power is useful because it's less sensitive
    to a channel simply being "louder" overall (e.g. due to electrode
    contact quality), so it's often more comparable across channels.
    """
    absolute = {}
    for band_name, band_range in BANDS.items():
        absolute[band_name] = band_power(signal_1d, sfreq, band_range)

    total = sum(absolute.values())
    relative = {}
    for band_name, power in absolute.items():
        relative[band_name] = (power / total) if total > 0 else 0.0

    return absolute, relative


def time_domain_features(signal_1d):
    """
    Simple statistical features describing the shape of the raw signal.

    mean:     average voltage. Should be close to 0 after filtering
              (filtering removes slow drift), so this is mostly a sanity
              check rather than an informative feature on its own.
    std:      standard deviation - how spread out the voltage values
              are. Higher std roughly means a "busier"/higher-amplitude
              signal.
    variance: std squared. Captures the same idea as std, included
              because some models/analyses prefer variance directly.
    rms:      root-mean-square, another measure of signal "size" that is
              always positive and closely related to signal power.
    """
    signal_1d = np.asarray(signal_1d)
    return {
        "mean": float(np.mean(signal_1d)),
        "std": float(np.std(signal_1d)),
        "variance": float(np.var(signal_1d)),
        "rms": float(np.sqrt(np.mean(signal_1d ** 2))),
    }


def hjorth_parameters(signal_1d):
    """
    Hjorth parameters: three classic EEG time-domain features computed
    from the signal and its first and second derivatives.

    activity:   variance of the signal itself. (Same idea as "variance"
                above - included here too since it's part of the
                standard Hjorth trio and some readers will expect it.)
    mobility:   a rough measure of the signal's dominant frequency -
                higher mobility means the signal changes more quickly
                from sample to sample.
    complexity: compares the mobility of the derivative to the mobility
                of the signal - roughly, how much the signal's frequency
                content changes over the window (how "irregular" it is).
    """
    signal_1d = np.asarray(signal_1d, dtype=float)
    first_deriv = np.diff(signal_1d)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal_1d)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0.0
    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0.0
    complexity = (mobility_d1 / mobility) if mobility > 0 else 0.0

    return {
        "hjorth_activity": float(activity),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def extract_features(window_data, sfreq, ch_names):
    """
    The main reusable function: takes one EEG window and returns a flat
    dictionary of named features, ready to become one row of our feature
    table.

    Parameters
    ----------
    window_data : numpy array, shape (n_channels, n_samples)
        The EEG segment to extract features from.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str, length n_channels
        Channel names, in the same order as window_data's rows. Used to
        build feature names like "Fp1_alpha_power".

    Returns
    -------
    dict mapping feature name -> value, e.g.:
        {
            "Fp1_alpha_power": ...,
            "Fp1_alpha_relative": ...,
            "Fp1_beta_power": ...,
            ...
            "Fp1_mean": ...,
            "Fp1_std": ...,
            "Fp1_hjorth_mobility": ...,
            ...
        }

    This function is deliberately kept independent of how the window was
    obtained, so we can reuse it unchanged later when processing live/
    replayed EEG for predictions.
    """
    features = {}

    for i, ch_name in enumerate(ch_names):
        signal_1d = window_data[i, :]

        absolute, relative = band_powers_for_channel(signal_1d, sfreq)
        for band_name in BANDS:
            features[f"{ch_name}_{band_name}_power"] = absolute[band_name]
            features[f"{ch_name}_{band_name}_relative"] = relative[band_name]

        for feat_name, value in time_domain_features(signal_1d).items():
            features[f"{ch_name}_{feat_name}"] = value

        for feat_name, value in hjorth_parameters(signal_1d).items():
            features[f"{ch_name}_{feat_name}"] = value

    return features