"""
baseline.py

Implements the "personalized baseline" concept from Section 9 of the
spec: instead of comparing anyone's EEG to a universal threshold, we
calculate how far the CURRENT window's features are from THIS
participant's OWN calibration-period statistics.

USAGE PATTERN:
    baseline = PersonalBaseline()
    # during the calibration period:
    baseline.add_calibration_sample(raw_features_dict)
    ... (repeat for every window during calibration) ...
    baseline.fit()
    # after calibration:
    deviations = baseline.deviation(raw_features_dict)  # -> z-scores
"""

import numpy as np


class PersonalBaseline:
    """Collects calibration-period features and computes z-score deviations."""

    def __init__(self):
        self._samples = []
        self._stats = None  # feature_name -> (mean, std)

    @property
    def is_fitted(self):
        return self._stats is not None

    @property
    def n_calibration_samples(self):
        return len(self._samples)

    def add_calibration_sample(self, raw_features):
        """Record one window's raw feature dict during the calibration period."""
        self._samples.append(raw_features)

    def fit(self):
        """
        Compute mean and standard deviation for every feature seen
        during calibration. Must be called once, after calibration
        ends and before deviation() is used.
        """
        if not self._samples:
            raise ValueError("No calibration samples collected - cannot fit baseline.")

        all_keys = set()
        for sample in self._samples:
            all_keys.update(sample.keys())

        stats = {}
        for key in all_keys:
            values = np.array([s[key] for s in self._samples if key in s])
            stats[key] = (float(values.mean()), float(values.std()))
        self._stats = stats

    def deviation(self, raw_features):
        """
        Return a dict of z-score deviations: how many standard
        deviations each current feature is from this participant's own
        calibration-period mean for that feature.

        A feature with baseline std effectively 0 (e.g. a constant
        signal during calibration) returns a deviation of 0.0 rather
        than dividing by a near-zero number, which would blow up into a
        meaningless huge value.
        """
        if not self.is_fitted:
            raise RuntimeError("Baseline not fitted yet - call fit() after calibration.")

        deviations = {}
        for key, value in raw_features.items():
            if key not in self._stats:
                continue
            mean, std = self._stats[key]
            deviations[key] = (value - mean) / std if std > 1e-12 else 0.0
        return deviations

    def summary(self):
        """Human-readable summary of the fitted baseline, for logging/printing."""
        if not self.is_fitted:
            return "Baseline not yet established."
        lines = [f"Baseline established from {self.n_calibration_samples} calibration windows:"]
        for key, (mean, std) in sorted(self._stats.items()):
            # Some raw features (e.g. absolute power, in volts^2) are
            # extremely small numbers - scientific notation keeps them
            # readable instead of printing as a misleading "0.00000".
            lines.append(f"  {key:28s} mean={mean:.3e}  std={std:.3e}")
        return "\n".join(lines)