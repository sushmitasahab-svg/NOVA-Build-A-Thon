"""
state_machine.py

Implements the READY / HIGH LOAD / PAUSE feedback state machine
(Sections 13, 14, 15, 29), with hysteresis so the state can't flicker:
a transition only happens once its triggering condition has been true
CONTINUOUSLY for a minimum duration, and returning to a better state
requires crossing a different (easier-to-satisfy-only-when-genuinely-
recovered) threshold than the one that triggered leaving it.

SIMPLIFICATION vs. the original spec, stated honestly:
    The spec describes an explicit READY -> HIGH LOAD -> PAUSE ->
    RECOVERY -> READY cycle with RECOVERY as its own persistent state.
    Here, "recovery" is treated as a labeled TRANSITION (from PAUSE
    back to READY) rather than a fourth state with its own timers -
    this keeps the state machine simpler while still letting the
    terminal output say "STATE STABILIZED" specifically when recovering
    from PAUSE, matching the spec's presentation-mode language.
"""

from src.neuroloop import config


class _ConditionTimer:
    """Tracks how long a boolean condition has been continuously true."""

    def __init__(self):
        self._true_since = None

    def update(self, is_true, current_time):
        if is_true:
            if self._true_since is None:
                self._true_since = current_time
            return current_time - self._true_since
        else:
            self._true_since = None
            return 0.0


class ReadinessStateMachine:
    """
    States: "READY", "HIGH_LOAD", "PAUSE".

    Call update(smoothed_score, current_time) once per window. Returns
    a dict describing the current state and whether it just changed.
    """

    def __init__(self):
        self.state = "READY"
        self._timer_below_high_load = _ConditionTimer()
        self._timer_below_pause = _ConditionTimer()
        self._timer_above_ready = _ConditionTimer()
        self._timer_above_pause_exit = _ConditionTimer()

    def update(self, smoothed_score, current_time):
        """
        Parameters
        ----------
        smoothed_score : float or None
            The current smoothed readiness score. If None (no reliable
            reading yet), the state machine does not transition.
        current_time : float
            Current time in seconds (from the EEG source), used to
            measure how long each threshold condition has persisted.

        Returns
        -------
        dict with:
            state : str, the state AFTER this update ("READY"/"HIGH_LOAD"/"PAUSE")
            changed : bool, True if the state changed this update
            previous_state : str
            recovered_from_pause : bool, True specifically on a
                PAUSE -> READY transition (for the "STATE STABILIZED" message)
        """
        # Always update all four timers, even if we don't act on all of
        # them in the current state - keeps their durations accurate
        # for whichever transition becomes relevant.
        dur_below_high_load = self._timer_below_high_load.update(
            smoothed_score is not None and smoothed_score < config.HIGH_LOAD_ENTER_THRESHOLD,
            current_time,
        )
        dur_below_pause = self._timer_below_pause.update(
            smoothed_score is not None and smoothed_score < config.PAUSE_ENTER_THRESHOLD,
            current_time,
        )
        dur_above_ready = self._timer_above_ready.update(
            smoothed_score is not None and smoothed_score > config.READY_RECOVER_THRESHOLD,
            current_time,
        )
        dur_above_pause_exit = self._timer_above_pause_exit.update(
            smoothed_score is not None and smoothed_score > config.PAUSE_EXIT_THRESHOLD,
            current_time,
        )

        previous_state = self.state
        recovered_from_pause = False

        if smoothed_score is None:
            # No reliable reading - never transition on missing/uncertain data.
            pass

        elif self.state == "READY":
            if dur_below_pause >= config.PAUSE_ENTER_MIN_DURATION_SEC:
                self.state = "PAUSE"
            elif dur_below_high_load >= config.HIGH_LOAD_ENTER_MIN_DURATION_SEC:
                self.state = "HIGH_LOAD"

        elif self.state == "HIGH_LOAD":
            if dur_below_pause >= config.PAUSE_ENTER_MIN_DURATION_SEC:
                self.state = "PAUSE"
            elif dur_above_ready >= config.READY_RECOVER_MIN_DURATION_SEC:
                self.state = "READY"

        elif self.state == "PAUSE":
            if dur_above_pause_exit >= config.PAUSE_EXIT_MIN_DURATION_SEC:
                self.state = "READY"
                recovered_from_pause = True

        return {
            "state": self.state,
            "changed": self.state != previous_state,
            "previous_state": previous_state,
            "recovered_from_pause": recovered_from_pause,
        }