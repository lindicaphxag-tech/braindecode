"""Independent synthetic MNE sign-parity checks for 18 canonical BIOT bipolar leads.

Tests the algebra behind an existing-module fix proposed as Braindecode #1577.
Does not run Braindecode ChannelLayer or any released pretrained checkpoint.
"""
import mne
import numpy as np
import pytest

PAIRS = (
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("C3", "A2"), ("C4", "A1"),
)


@pytest.mark.parametrize("anode,cathode", PAIRS, ids=[f"{a}-{b}" for a, b in PAIRS])
def test_reverse_derived_leads_are_exact_negatives(anode, cathode):
    rng = np.random.default_rng(20261118)
    data = rng.normal(size=(2, 257)) * 1e-5
    info = mne.create_info([anode, cathode], 256.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    ab = mne.set_bipolar_reference(
        raw, anode, cathode, ch_name=f"{anode}-{cathode}", copy=True, verbose=False
    )
    ba = mne.set_bipolar_reference(
        raw, cathode, anode, ch_name=f"{cathode}-{anode}", copy=True, verbose=False
    )
    first = ab.get_data(picks=[f"{anode}-{cathode}"])
    reversed_ = ba.get_data(picks=[f"{cathode}-{anode}"])
    np.testing.assert_array_equal(first, -reversed_)
    assert not np.array_equal(first, reversed_)


def test_bipolar_label_is_not_a_general_calibration_certificate():
    # Different reference offsets cancel for simultaneous voltage differences,
    # but unknown gains and acquisition differences cannot be inferred by name.
    x, y, ref1, ref2 = 1.0, -0.5, 0.1, -0.3
    ab = (x - ref1) - (y - ref1)
    ba = (y - ref2) - (x - ref2)
    assert ab == -ba
    assert ab != -(2.0 * ba)
