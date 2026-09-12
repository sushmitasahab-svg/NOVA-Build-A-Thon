"""
config.py

All tunable NeuroLoop parameters live here, in one place, so nothing is
hardcoded elsewhere in the system. Change a threshold here, not by
hunting through multiple files.
"""

# ---- Windowing (Section 5 of the spec) ----
WINDOW_LENGTH_SEC = 4.0   # how much EEG history each state estimate looks at
STEP_SEC = 1.0            # how often we recompute the state
# NOTE: the original spec suggested a 250-500ms update interval. We use
# 1.0s here deliberately - fast enough to feel responsive in a terminal
# demo, slow enough that every step produces a clean, readable line of
# output rather than a flood of near-duplicate prints.

# ---- Channels of interest (Section 8) ----
# Chosen to match the Unicorn Hybrid Black's 8-channel montage exactly,
# so this code runs unchanged whether we're replaying an ANT Neuro file
# (which happens to include all these channel names too) or reading
# live from the actual Unicorn headset tomorrow.
CHANNELS_OF_INTEREST = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
FRONTAL_CHANNEL = "Fz"
CENTRAL_CHANNELS = ["C3", "Cz", "C4"]
POSTERIOR_CHANNELS = ["Pz", "PO7", "Oz", "PO8"]

# ---- Calibration (Section 9) ----
# The spec recommends 60-120s for a real chess session. We default to
# 20.0s here ONLY for tonight's Eyes-Open/Closed replay test, because
# that recording's first "Open" block is exactly 20s long and is a
# genuinely calm, stable period - a reasonable stand-in baseline for
# testing. This must be changed back to 60-120s for the real chess
# recording.
CALIBRATION_DURATION_SEC = 20.0

# ---- Signal quality (Section 6) ----
# A channel's window is considered "usable" if its standard deviation
# falls in this range (volts). Below the low end likely means a flat/
# disconnected channel; above the high end likely means saturation or
# a large artifact.
QUALITY_MIN_STD_VOLTS = 1e-8
QUALITY_MAX_STD_VOLTS = 200e-6
QUALITY_THRESHOLD = 0.75  # minimum fraction of usable channels to trust the reading

# ---- Readiness thresholds and hysteresis (Sections 12-14) ----
# Entering a worse state requires readiness to drop below the "enter"
# threshold and STAY there for the minimum duration. Returning to a
# better state requires readiness to rise above the (higher) "exit"
# threshold and stay there for its own minimum duration. Using
# different thresholds to enter vs. exit each state (hysteresis) is
# what stops the system from flickering back and forth.
HIGH_LOAD_ENTER_THRESHOLD = 85
HIGH_LOAD_ENTER_MIN_DURATION_SEC = 2.0
READY_RECOVER_THRESHOLD = 90
READY_RECOVER_MIN_DURATION_SEC = 3.0
PAUSE_ENTER_THRESHOLD = 45
PAUSE_ENTER_MIN_DURATION_SEC = 3.0
PAUSE_EXIT_THRESHOLD = 65
PAUSE_EXIT_MIN_DURATION_SEC = 5.0

# ---- Temporal smoothing (Section 13) ----
# Exponential moving average factor applied to the raw readiness score
# before it's handed to the state machine. Smaller = smoother/slower
# to react, larger = more responsive/more jittery.
SMOOTHING_ALPHA = 0.3

# ---- Readiness score mapping (Section 12) ----
# Converts a combined baseline-deviation magnitude into a 0-100 score:
#   readiness = 100 * exp(-READINESS_DECAY_RATE * combined_deviation)
# combined_deviation is the RMS (root-mean-square) of all feature
# z-scores for a window - roughly "how many standard deviations away
# from personal baseline, overall." A deviation of 0 gives 100. Larger
# deviations decay smoothly toward 0. This rate is a reasonable
# starting point, NOT a validated clinical threshold - per the spec,
# these numbers should eventually be tuned/learned from real data.
READINESS_DECAY_RATE = 0.15