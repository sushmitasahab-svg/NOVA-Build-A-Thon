"""
live_eeg_source.py

The live counterpart to ReplayEEGSource: connects to a real, currently-
broadcasting LSL (Lab Streaming Layer) stream - e.g. from ANT Neuro's
recording software during tomorrow's live demo - and implements the
exact same EEGSource interface (ch_names, sfreq, current_time,
get_window, advance).

Because it implements the same interface, NOTHING downstream (state_
vector.py, baseline.py, readiness.py, smoothing.py, state_machine.py,
feedback_cli.py, music_controller.py) needs to change to go from
replaying a file to reading a live headset. run_neuroloop_demo.py only
needs to construct a LiveEEGSource instead of a ReplayEEGSource.

HOW CONNECTING WORKS (per the pylsl library's own documented API):
    1. resolve_byprop('type', 'EEG', ...) searches the network for a
       broadcasting EEG stream and waits up to a timeout for one to
       appear.
    2. StreamInlet(stream_info) opens a connection to it.
    3. inlet.pull_chunk() repeatedly asks "what new samples have
       arrived since I last checked" - this is how advance() gets new
       data.
    4. inlet.info().desc() can contain channel labels as metadata, IF
       the broadcasting software included them (confirmed common
       practice for professional EEG systems, but not guaranteed) - we
       try to read real labels first, and only fall back to a manually
       provided list if the stream doesn't supply them.

WHAT'S DIFFERENT FROM ReplayEEGSource:
    - advance() actually WAITS for real time to pass (it blocks until
      new samples arrive or a timeout is hit) - it cannot "instantly"
      skip through a live stream the way replay can skip through a
      file, because live data only exists once the headset produces it.
    - current_time is based on the LSL stream's own real timestamps,
      not a manually incremented counter.
    - There is no fixed "recording_duration" - advance() keeps
      returning True indefinitely as long as the connection is alive.

TESTED VIA: a local simulated LSL broadcast built from a real recording
(see the accompanying rehearsal script) - this exercises the full
resolve -> connect -> pull_chunk -> buffer -> window path against real
brain-signal values, without needing the physical headset. It has NOT
been tested against ANT Neuro's actual live software output, since
that requires the physical demo setup.
"""

import time
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from src.neuroloop.eeg_source import EEGSource


class LiveEEGSource(EEGSource):
    def __init__(self, stream_type="EEG", fallback_channel_names=None,
                 connect_timeout_sec=15.0, buffer_seconds=15.0):
        print(f"Looking for a live LSL stream of type '{stream_type}'...")
        streams = resolve_byprop("type", stream_type, timeout=connect_timeout_sec)
        if not streams:
            raise RuntimeError(
                f"No LSL stream of type '{stream_type}' was found within "
                f"{connect_timeout_sec}s. Make sure the EEG software is "
                f"running and broadcasting over LSL."
            )

        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        self._sfreq = info.nominal_srate()
        self._ch_names = self._read_channel_labels(info, fallback_channel_names)
        n_channels = len(self._ch_names)
        print(f"Connected. Channels: {self._ch_names}  Sampling rate: {self._sfreq} Hz")

        max_buffer_samples = int(buffer_seconds * self._sfreq)
        self._sample_buffer = deque(maxlen=max_buffer_samples)   # each item: array of n_channels values
        self._timestamp_buffer = deque(maxlen=max_buffer_samples)

        self._current_time = 0.0
        self._n_channels = n_channels
        self._start_timestamp = None  # set on first received sample - see advance()

    @staticmethod
    def _read_channel_labels(info, fallback_channel_names):
        """
        Try to read real channel names from the stream's own metadata.
        Falls back to a manually provided list if the stream doesn't
        supply usable labels (some LSL outlets omit this metadata).
        """
        try:
            n = info.channel_count()
            ch = info.desc().child("channels").child("channel")
            names = []
            for _ in range(n):
                names.append(ch.child_value("label"))
                ch = ch.next_sibling()
            if len(names) == n and all(names):
                return names
        except Exception:
            pass

        if fallback_channel_names:
            print("NOTE: stream did not provide channel labels - using the "
                  "provided fallback channel name list instead.")
            return list(fallback_channel_names)

        raise RuntimeError(
            "The LSL stream did not provide channel name metadata, and no "
            "fallback_channel_names was given. Pass fallback_channel_names="
            "[...] with the known montage order to proceed."
        )

    @property
    def ch_names(self):
        return self._ch_names

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def current_time(self):
        return self._current_time

    def advance(self, step_sec):
        """
        Block for up to step_sec seconds, pulling in whatever new
        samples arrive from the live stream during that time.
        """
        deadline = time.time() + step_sec
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            chunk, timestamps = self._inlet.pull_chunk(timeout=remaining,
                                                         max_samples=int(self._sfreq))
            if timestamps:
                if self._start_timestamp is None:
                    # LSL timestamps are relative to an arbitrary local
                    # clock (they do NOT start at 0) - anchor our own
                    # "current_time" to the first sample we ever see, so
                    # calibration/step logic (which expects time to
                    # start near zero) behaves correctly.
                    self._start_timestamp = timestamps[0]
                for row, ts in zip(chunk, timestamps):
                    # UnicornLSL broadcasts in microvolts, but our pipeline
                    # assumes volts everywhere else - convert here.
                    row_volts = [v * 1e-6 for v in row]
                    self._sample_buffer.append(row_volts)
                    self._timestamp_buffer.append(ts)
                self._current_time = timestamps[-1] - self._start_timestamp

        return True  # a live connection is considered "always more data" until it errors

    def get_window(self, window_length_sec):
        """
        Return the most recent window_length_sec seconds of buffered
        samples, or None if not enough has been buffered yet.
        """
        if not self._timestamp_buffer:
            return None

        # Compare in the same (raw LSL timestamp) space the buffer is
        # stored in - self._current_time is elapsed-since-start, so we
        # rebuild the equivalent raw cutoff here rather than mixing units.
        latest_raw = self._timestamp_buffer[-1]
        cutoff_raw = latest_raw - window_length_sec
        selected = [s for s, t in zip(self._sample_buffer, self._timestamp_buffer)
                    if t >= cutoff_raw]

        min_required = int(window_length_sec * self._sfreq * 0.8)  # allow some tolerance
        if len(selected) < min_required:
            return None

        # shape (n_samples, n_channels) -> (n_channels, n_samples), matching
        # the same convention ReplayEEGSource.get_window() already uses.
        return np.array(selected).T