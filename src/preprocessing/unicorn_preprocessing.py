"""
unicorn_preprocessing.py

Loads a g.tec Unicorn Hybrid Black recording (exported as CSV) into an
MNE Raw object, ready for the same downstream code (state_vector.py,
baseline.py, etc.) that already works with our ANT Neuro recordings.

WHY THIS IS A SEPARATE FILE FROM eeg_preprocessing.py:
    eeg_preprocessing.py is built specifically around ANT Neuro's .cnt
    format (via mne.io.read_raw_ant). The Unicorn exports .bdf/.csv
    instead, with a different channel layout and, importantly, a real
    bug we found in this specific export: the BDF file's unit label
    ("uV") gets garbled in a way that makes MNE's automatic BDF reader
    misapply the scaling by a factor of 1,000,000. We sidestep that
    entirely by reading the CSV export directly (already in correct
    real-world uV values, confirmed by manual inspection) and building
    the MNE Raw object ourselves with the correct, known scaling.

IMPORTANT ASSUMPTION - STATED EXPLICITLY, NOT HIDDEN:
    The Unicorn's raw CSV only labels channels generically ("EEG 1"
    through "EEG 8"), not with real 10-20 names. We assume they are in
    g.tec's documented DEFAULT order: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
    This has NOT been independently confirmed against this specific
    headset's configuration - if your team's Unicorn Suite config uses
    a different channel order, this mapping would need to be corrected.
"""

import numpy as np
import mne

from src.preprocessing.eeg_preprocessing import BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ

UNICORN_SFREQ = 250.0

# g.tec's documented default 8-channel montage, in order - see the
# caveat above. If this project later confirms a different order from
# the Unicorn Suite config, update this list (nothing else needs to change).
UNICORN_DEFAULT_CHANNEL_ORDER = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def load_and_preprocess_unicorn_csv(csv_path, verbose=False):
    """
    Load a Unicorn CSV export and apply the same minimal preprocessing
    as eeg_preprocessing.load_and_preprocess (bandpass filter only).

    Returns an MNE Raw object with 8 channels named per
    UNICORN_DEFAULT_CHANNEL_ORDER, sampled at 250 Hz.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    eeg_cols = [c for c in df.columns if c.startswith("EEG")]
    if len(eeg_cols) != len(UNICORN_DEFAULT_CHANNEL_ORDER):
        raise ValueError(
            f"Expected {len(UNICORN_DEFAULT_CHANNEL_ORDER)} EEG channels, "
            f"found {len(eeg_cols)} columns starting with 'EEG' in {csv_path}"
        )

    # CSV values are already in real-world microvolts (confirmed by manual
    # inspection). MNE's internal convention is volts, so divide by 1e6.
    data_uv = df[eeg_cols].to_numpy().T  # shape (n_channels, n_samples)
    data_volts = data_uv * 1e-6

    info = mne.create_info(
        ch_names=UNICORN_DEFAULT_CHANNEL_ORDER,
        sfreq=UNICORN_SFREQ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data_volts, info, verbose="error" if not verbose else None)

    raw.filter(l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ,
               verbose="error" if not verbose else None)

    return raw