"""
eeg_preprocessing.py

Minimal, explainable preprocessing for our ANT Neuro .cnt recordings.

What we do here, and why:
1. Load the raw .cnt file with MNE (the standard Python library for EEG).
2. Drop channels we already know are not real signal:
       M1 and M2 (mastoid electrodes) were not connected to anything
       during recording, per notes.docx from ANT Neuro. Keeping them
       would just feed noise into our features.
3. Apply a bandpass filter (0.5-40 Hz).
       EEG frequency bands we care about (delta through beta) all live
       inside 0.5-40 Hz. Filtering removes slow baseline drift (below
       0.5 Hz) and high-frequency noise/muscle artifact (above 40 Hz)
       without touching the signal we actually want to analyze.

We deliberately do NOT do anything fancier yet (no ICA, no advanced
artifact rejection) - this is the minimum necessary step, matching the
"start simple" instruction. If we find the features look unreasonable
later, we will revisit this file and add more.
"""

import mne

DEAD_CHANNELS = ["M1", "M2"]
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0


def load_and_preprocess(cnt_path, verbose=False):
    """
    Load a .cnt file and apply our minimal preprocessing.

    Returns an MNE Raw object with:
      - M1/M2 dropped
      - 0.5-40 Hz bandpass filter applied
      - data loaded into memory (preload=True), since we need to slice
        it into per-trial windows later
    """
    raw = mne.io.read_raw_ant(cnt_path, preload=True,
                               verbose="error" if not verbose else None)

    # Drop dead mastoid channels if present (some recordings may not have them
    # depending on montage - we check first so this never errors out).
    channels_to_drop = [ch for ch in DEAD_CHANNELS if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    # Bandpass filter. Keeps 0.5-40 Hz, which covers delta through beta.
    raw.filter(l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ,
               verbose="error" if not verbose else None)

    return raw