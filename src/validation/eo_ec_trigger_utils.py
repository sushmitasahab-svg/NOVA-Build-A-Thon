"""
eo_ec_trigger_utils.py

Parses the Eyes Open/Eyes Closed .trg file into a table of blocks
(one row per "Open" or "Closed" segment).

WHERE THE CODE MEANINGS CAME FROM:
    Unlike the Flip Cup dataset (which had an explicit Marker
    definitions.xlsx), the EO/EC trigger codes were not documented in a
    plain-language file. We determined their meaning by cross-referencing
    the .trg trigger codes against data/raw/eyes_open_closed/
    brainstory_project.bst - a saved analysis project file that contains
    an "Edit Markers" step explicitly labeling:
        code 11 -> 12  =  "Open"    (eyes open block, 20.0s each)
        code 13 -> 14  =  "Closed"  (eyes closed block, 20.0s each)
    We are not guessing this mapping - it is read directly from that
    project file's own labels.

    Other codes seen in the trigger file (8, 9, 62, 63) are NOT block
    boundaries - 62/63 mark experiment start/stop, and 8/9 are recurring
    sub-markers within blocks whose exact purpose the project file does
    not explain. We do not use 8/9 for anything here.
"""

import pandas as pd

OPEN_START_CODE = 11
OPEN_END_CODE = 12
CLOSED_START_CODE = 13
CLOSED_END_CODE = 14


def load_trigger_file(trg_path):
    """Same format as the Flip Cup .trg files: time_s, sample, code per line."""
    df = pd.read_csv(trg_path, sep=r"\s+", header=None,
                      names=["time_s", "sample", "code"])
    df = df.dropna(subset=["code"])
    df = df[pd.to_numeric(df["code"], errors="coerce").notna()].copy()
    df["code"] = df["code"].astype(int)
    return df.reset_index(drop=True)


def build_block_table(trg_df):
    """
    Turn the trigger codes into one row per Open/Closed block, in the
    order they occur in the recording.
    """
    blocks = []
    block_id = 0

    for start_code, end_code, label in [
        (OPEN_START_CODE, OPEN_END_CODE, "Open"),
        (CLOSED_START_CODE, CLOSED_END_CODE, "Closed"),
    ]:
        starts = trg_df.loc[trg_df["code"] == start_code, "time_s"].reset_index(drop=True)
        ends = trg_df.loc[trg_df["code"] == end_code, "time_s"].reset_index(drop=True)
        n = min(len(starts), len(ends))
        for i in range(n):
            blocks.append({
                "block_id": None,  # filled in after sorting by time
                "condition": label,
                "start_time": starts[i],
                "end_time": ends[i],
                "duration": ends[i] - starts[i],
            })

    blocks_df = pd.DataFrame(blocks).sort_values("start_time").reset_index(drop=True)
    blocks_df["block_id"] = blocks_df.index
    return blocks_df