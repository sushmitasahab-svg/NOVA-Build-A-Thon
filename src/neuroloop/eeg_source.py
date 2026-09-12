"""
eeg_source.py

Defines a common interface for "where EEG comes from," so the rest of
the NeuroLoop system (baseline, features, readiness, feedback) never
needs to know or care whether the data is:
  - a pre-recorded file being replayed as if live (ReplayEEGSource)
  - a real live headset stream (a future LiveEEGSource)
  - synthetic test data (a future SimulatedEEGSource)

WHY THIS MATTERS FOR TONIGHT:
    We don't have a chess+EEG recording yet, and we won't have live
    headset access until tomorrow's demo. ReplayEEGSource lets us build
    and fully test the ENTIRE rest of the pipeline right now, using EEG
    recordings we already trust (starting with Eyes Open/Closed), by
    pretending the file is streaming in live, one small time-step at a
    time. When the real headset is ready, only a new EEGSource subclass
    needs to be written - nothing else in the system changes.

HOW REPLAY WORKS:
    ReplayEEGSource holds an already-preprocessed MNE Raw object and a
    "current_time" pointer. Each call to advance() moves that pointer
    forward by a small step (e.g. 0.5s), simulating the passage of real
    time. get_window() returns the most recent N seconds of data ending
    at the current pointer - never data from the future, which matches
    how a real live system would have to behave.
"""

from abc import ABC, abstractmethod


class EEGSource(ABC):
    """Common interface every EEG data source must implement."""

    @property
    @abstractmethod
    def ch_names(self):
        """List of channel names, in the order data rows are returned."""
        raise NotImplementedError

    @property
    @abstractmethod
    def sfreq(self):
        """Sampling frequency in Hz."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_time(self):
        """Current position in seconds since the source started."""
        raise NotImplementedError

    @abstractmethod
    def get_window(self, window_length_sec):
        """
        Return the most recent `window_length_sec` seconds of EEG data,
        ending at current_time, as a numpy array of shape
        (n_channels, n_samples).

        Returns None if not enough history exists yet (e.g. right at
        the very start, before current_time >= window_length_sec) -
        callers must handle this, exactly as a real live system would
        have to wait before it has a full window.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self, step_sec):
        """
        Move current_time forward by step_sec.

        Returns True if there is still more data available after
        advancing, False if the source has run out (e.g. reached the
        end of a replayed recording).
        """
        raise NotImplementedError


class ReplayEEGSource(EEGSource):
    """
    Replays an already-preprocessed MNE Raw object as if it were a live
    stream, one time-step at a time.

    Parameters
    ----------
    raw : mne.io.Raw
        An MNE Raw object that has ALREADY been preprocessed (e.g. via
        src/preprocessing/eeg_preprocessing.py's load_and_preprocess).
        This class does not filter or clean the data - it only controls
        the flow of time.
    start_time : float
        Where in the recording to start "playing" from (seconds).
        Defaults to 0.0 (the very beginning of the recording).
    """

    def __init__(self, raw, start_time=0.0):
        self._raw = raw
        self._ch_names = raw.ch_names
        self._sfreq = raw.info["sfreq"]
        self._recording_duration = raw.n_times / self._sfreq
        self._current_time = start_time

    @property
    def ch_names(self):
        return self._ch_names

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def current_time(self):
        return self._current_time

    @property
    def recording_duration(self):
        """Total length of the underlying recording, in seconds."""
        return self._recording_duration

    def get_window(self, window_length_sec):
        window_start = self._current_time - window_length_sec
        if window_start < 0:
            # Not enough history yet - same situation a live system
            # would face right after startup, before enough time has
            # passed to fill one full window.
            return None
        if self._current_time > self._recording_duration:
            return None

        start_idx = self._raw.time_as_index(window_start)[0]
        end_idx = self._raw.time_as_index(self._current_time)[0]
        if end_idx <= start_idx:
            return None

        return self._raw.get_data(start=start_idx, stop=end_idx)

    def advance(self, step_sec):
        self._current_time += step_sec
        return self._current_time <= self._recording_duration