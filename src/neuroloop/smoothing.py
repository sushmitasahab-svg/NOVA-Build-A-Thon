"""
smoothing.py

Applies temporal smoothing to the raw, window-by-window readiness score
(Section 13 of the spec), so a single noisy window can't cause the
displayed state to flicker.

Also implements the Section 6 rule that bad signal quality should never
be silently interpreted as "low readiness": when a window is flagged
unreliable, we FREEZE the smoothed score at its last known good value
instead of feeding a possibly-meaningless number into the average.
"""

from src.neuroloop import config


class ReadinessSmoother:
    """Exponential moving average over raw readiness scores."""

    def __init__(self, alpha=None):
        self._alpha = alpha if alpha is not None else config.SMOOTHING_ALPHA
        self._smoothed_value = None

    @property
    def value(self):
        """Current smoothed readiness score, or None if never updated."""
        return self._smoothed_value

    def update(self, raw_score, reliable):
        """
        Feed in one window's raw readiness score.

        If reliable is False (poor signal quality), the smoothed value
        is left UNCHANGED (frozen) and returned as-is - we do not let a
        bad-quality window pull the score down, per Section 6.

        Returns the current smoothed value (float), or None if no
        reliable reading has ever been seen yet.
        """
        if not reliable:
            return self._smoothed_value

        if self._smoothed_value is None:
            self._smoothed_value = raw_score
        else:
            self._smoothed_value = (
                self._alpha * raw_score + (1 - self._alpha) * self._smoothed_value
            )
        return self._smoothed_value