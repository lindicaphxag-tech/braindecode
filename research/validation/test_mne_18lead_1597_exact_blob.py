"""MNE 18-bipolar-lead oracle for the *exact Git blobs* of #1593 and #1597.

Run from a full checkout after fetching the PR branch:
  git fetch origin fix/exact-derivation-empty-and-case-20261119
  BD_CHANNELS_REF=merged python -m pytest -q research/validation/test_mne_18lead_1597_exact_blob.py
  BD_CHANNELS_REF=followup python -m pytest -q research/validation/test_mne_18lead_1597_exact_blob.py

Baseline: 10 pass / 2 fail. Proposed: 12/12 pass. PyTorch CPU + MNE 1.11.
"""
import hashlib
import os
import subprocess
import types
import numpy as np
import pytest
import mne
import torch

REFS = {
    "merged": (
        "9c541453e2c13ce0c3680a1582a136ca121b0da7",
        "896f80ac2934851a3ef7ef731c83ac43be0aa845",
    ),
    "followup": (
        "baf8e4c734181727a34a33df126a89a356f0cc81",
        "3b05938f993e7fd6f1e6651614018129dbb8defc",
    ),
}
choice = os.environ.get("BD_CHANNELS_REF", "followup")
if choice not in REFS:
    raise ValueError("BD_CHANNELS_REF must be merged or followup")
commit, expected_blob = REFS[choice]
raw = subprocess.check_output(["git", "show", f"{commit}:braindecode/functional/channels.py"])
actual_blob = hashlib.sha1(f"blob {len(raw)}\\0".encode() + raw).hexdigest()
assert actual_blob == expected_blob, f"unexpected original Git blob: {actual_blob}"
module = types.ModuleType("bd1597_mne_exact_channels")
exec(compile(raw, f"{commit}:braindecode/functional/channels.py", "exec"), module.__dict__)

PAIRS = [
    ("Fp1","F7"),("F7","T7"),("T7","P7"),("P7","O1"),
    ("Fp1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),
    ("Fp2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
    ("Fp2","F8"),("F8","T8"),("T8","P8"),("P8","O2"),
    ("Fz","Cz"),("Cz","Pz"),
]
ELECTRODES = list(dict.fromkeys(x for pair in PAIRS for x in pair))
TARGETS = [f"{a}-{b}" for a,b in PAIRS]
INPUTS = [f"{name}-REF" for name in ELECTRODES]

@pytest.mark.parametrize("seed", [0,1,2,3,4])
def test_18_bipolar_channels_against_independent_mne(seed):
    rng = np.random.default_rng(seed)
    potentials = rng.normal(size=(len(ELECTRODES), 768)) * 1e-5
    shared_reference = rng.normal(size=(1, 768)) * 2e-5
    referenced = potentials - shared_reference
    info = mne.create_info(ch_names=ELECTRODES, sfreq=256.0, ch_types=["eeg"]*len(ELECTRODES))
    raw = mne.io.RawArray(referenced, info, verbose="error")
    mne_bipolar = mne.set_bipolar_reference(
        raw, anode=[a for a,b in PAIRS], cathode=[b for a,b in PAIRS],
        ch_name=TARGETS, drop_refs=False, copy=True, verbose="error",
    )
    expected = mne_bipolar.get_data(picks=TARGETS)
    W, exact = module.exact_derivation(TARGETS, INPUTS)
    assert W.shape == (18, len(INPUTS))
    assert W.dtype == torch.float64 and exact.tolist() == [True] * 18
    actual = (W @ torch.tensor(referenced, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
    assert np.max(np.abs(actual - expected)) < 1e-16

@pytest.mark.parametrize("seed", [11,12,13,14,15])
def test_reverse_polarity_with_redundant_bipolar_chains(seed):
    rng = np.random.default_rng(seed)
    reverse_inputs = [f"{b}-{a}" for a,b in PAIRS]
    W, exact = module.exact_derivation(TARGETS, reverse_inputs)
    assert W.shape == (18,18) and exact.tolist() == [True] * 18
    R, electrodes = module.derivation_matrix(reverse_inputs)
    T, electrodes = module.derivation_matrix(TARGETS, electrodes)
    R = torch.nn.functional.pad(R, (0, T.shape[1] - R.shape[1]))
    torch.testing.assert_close(W @ R, T, atol=1e-10, rtol=0)
    v = torch.tensor(rng.normal(size=(len(electrodes), 512)), dtype=torch.float64)
    torch.testing.assert_close(W @ (R @ v), T @ v, atol=1e-10, rtol=0)
    # W need not equal -I: chain dependencies permit a distinct min-norm W.
    assert not np.allclose(W.numpy(), -np.eye(18), atol=1e-12, rtol=0)

@pytest.mark.parametrize("names,reference", [
    (["Fp1","Cz"],"Cz"), (["fP1","cZ"],"cZ"),
])
def test_case_normalized_explicit_reference_columns(names, reference):
    R, electrodes = module.derivation_matrix(
        ["fP1"], electrodes=names, reference=reference,
    )
    assert electrodes == ["fp1","cz"]
    assert R.tolist() == [[1.0,-1.0]]
