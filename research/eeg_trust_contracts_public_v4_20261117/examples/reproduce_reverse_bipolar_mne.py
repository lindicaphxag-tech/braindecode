"""Synthetic MNE parity: a measured reverse bipolar channel needs -1, not +1.

No Braindecode or published EEG checkpoint is imported by this demonstration.
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np


def main():
    rng = np.random.default_rng(20261118)
    data = rng.standard_normal((2, 256)) * 1e-5
    info = mne.create_info(["F3", "C3"], sfreq=256, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(data, info, verbose=False)
    forward = mne.set_bipolar_reference(
        raw, anode="F3", cathode="C3", ch_name="F3-C3", copy=True, verbose=False
    ).get_data(picks=["F3-C3"])
    reverse = mne.set_bipolar_reference(
        raw, anode="C3", cathode="F3", ch_name="C3-F3", copy=True, verbose=False
    ).get_data(picks=["C3-F3"])
    np.testing.assert_array_equal(forward, -reverse)
    pos = info.get_montage().get_positions()["ch_pos"]
    result = {
        "test": "measured reverse bipolar matches negative of requested target",
        "seed": 20261118,
        "mne_version": mne.__version__,
        "max_abs_error_signed_conversion_V": float(np.max(np.abs(forward + reverse))),
        "max_abs_error_wrong_positive_copy_V": float(np.max(np.abs(forward - reverse))),
        "same_midpoint_m": ((pos["F3"] + pos["C3"]) / 2).tolist(),
        "scope": "synthetic MNE algebra; not Braindecode full-suite or EEG performance",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
