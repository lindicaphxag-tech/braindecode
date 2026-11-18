"""Independent exact-blob PyTorch CPU regression after Braindecode #1593 -> #1597.

On the fork branch evidence/1597-exact-head-redgreen-20261119:
    BD_CHANNELS_REF=merged pytest -q research/validation/test_derivation_1597_redgreen.py
    BD_CHANNELS_REF=followup pytest -q research/validation/test_derivation_1597_redgreen.py

No source rewrite: git-show extracts the actual upstream Git blob. The SHA-1
assertion pins its full byte contents. 31 cases, with a new #1593 reference-argument backward-compatibility check.
This is NOT the full upstream Braindecode test suite or official CI.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
import types
import pytest
import torch

SOURCES = {
    "merged": ("9c541453e2c13ce0c3680a1582a136ca121b0da7",
               "896f80ac2934851a3ef7ef731c83ac43be0aa845"),
    "followup": ("baf8e4c734181727a34a33df126a89a356f0cc81",
                 "3b05938f993e7fd6f1e6651614018129dbb8defc"),
}
CHOICE = os.environ.get("BD_CHANNELS_REF", "followup")
if CHOICE not in SOURCES:
    raise ValueError("BD_CHANNELS_REF must be 'merged' or 'followup'")
commit, expected_blob = SOURCES[CHOICE]
raw = subprocess.check_output(["git", "show", f"{commit}:braindecode/functional/channels.py"])
actual_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
assert actual_blob == expected_blob, f"source mismatch {actual_blob} != {expected_blob}"
mod = types.ModuleType("exact_source_channels")
exec(compile(raw, f"{commit}:braindecode/functional/channels.py", "exec"), mod.__dict__)


def test_source_is_sha_pinned():
    assert actual_blob == expected_blob


@pytest.mark.parametrize("inputs", [[], ["F3-C3"]])
def test_empty_target_result_shapes(inputs):
    W, exact = mod.exact_derivation([], inputs)
    assert W.shape == (0, len(inputs)) and W.dtype == torch.float64
    assert exact.shape == (0,) and exact.dtype == torch.bool


def test_user_supplied_electrode_case_normalization():
    R, names = mod.derivation_matrix(["F3-C3"], ["F3", "C3"])
    assert names == ["f3", "c3"]
    assert R.tolist() == [[1.0, -1.0]]


def test_case_insensitive_duplicate_columns():
    R, names = mod.derivation_matrix(["f3-C3"], ["F3", "f3", "c3", "C3"])
    assert names == ["f3", "c3"]
    assert R.tolist() == [[1.0, -1.0]]


@pytest.mark.parametrize("targets, inputs, weights", [
    (["F3-C3"], ["C3-F3"], [[-1.0]]),
    (["F3-P3"], ["F3-C3", "C3-P3"], [[1.0, 1.0]]),
    (["F3-C3"], ["F3", "C3"], [[1.0, -1.0]]),
    (["F3-C3"], ["F3-REF", "C3-REF"], [[1.0, -1.0]]),
    (["P3"], ["C3", "C3-P3"], [[1.0, -1.0]]),
    (["F3-C3"], ["F3-C3", "F3-C3"], [[0.5, 0.5]]),
    (["F3-C3"], ["F3-R1", "C3-R2"], None),
    (["F3"], ["F3-C3"], None),
    (["F3-C3"], [], None),
])
def test_preexisting_merged_exact_contract(targets, inputs, weights):
    W, exact = mod.exact_derivation(targets, inputs)
    assert exact.tolist() == [weights is not None]
    assert W.shape == (len(targets), len(inputs))
    assert W.tolist() == (weights or [[0.0] * len(inputs)])


def test_simultaneous_targets_with_one_nonidentifiable():
    W, exact = mod.exact_derivation(["F3-C3", "F3-P3", "F3"], ["F3-C3", "C3-P3"])
    assert exact.tolist() == [True, True, False] and W.shape == (3, 2)
    torch.testing.assert_close(W[:2], torch.tensor([[1,0],[1,1]], dtype=torch.float64),
                               rtol=0, atol=1e-10)
    assert torch.count_nonzero(W[2]) == 0


def test_torch_signal_gradient_preserved():
    W, exact = mod.exact_derivation(["F3-C3"], ["C3-F3"])
    assert exact.tolist() == [True]
    x = torch.randn((1, 2, 11), dtype=torch.float64, requires_grad=True)
    y = W.matmul(x[:,0:1,:])
    torch.testing.assert_close(y, -x[:,0:1,:], rtol=0, atol=0)
    y.sum().backward()
    assert torch.equal(x.grad[:,0:1,:], -torch.ones_like(x[:,0:1,:]))
    assert torch.count_nonzero(x.grad[:,1:2,:]) == 0


@pytest.mark.parametrize("seed", list(range(12)))
def test_random_electrode_potential_reconstruction(seed):
    generator = torch.Generator().manual_seed(seed)
    inputs = ["F3-C3", "C3-P3", "F3-REF", "C3-REF"]
    targets = ["F3-P3", "P3-C3", "F3-C3"]
    R, names = mod.derivation_matrix(inputs)
    T, names = mod.derivation_matrix(targets, names)
    R = torch.nn.functional.pad(R, (0, T.shape[1] - R.shape[1]))
    v = torch.randn((len(names), 15), generator=generator, dtype=torch.float64)
    W, exact = mod.exact_derivation(targets, inputs)
    assert exact.tolist() == [True, True, True]
    torch.testing.assert_close(W @ (R @ v), T @ v, atol=1e-9, rtol=0)


def test_channel_name_case_insensitivity():
    W, exact = mod.exact_derivation(["Fp1-F7"], ["f7-fP1"])
    assert exact.tolist() == [True] and W.tolist() == [[-1.0]]


def test_nonexact_target_weight_is_zero():
    W, exact = mod.exact_derivation(["F3-C3", "F3"], ["C3-F3"])
    assert exact.tolist() == [True, False] and W.tolist() == [[-1.0], [0.0]]


def test_latest_reference_argument_with_explicit_mixed_case_columns():
    # Added for the upstream #1593 reference-aware derivation_matrix signature.
    R, electrodes = mod.derivation_matrix(
        ["F3"], electrodes=["F3", "Cz"], reference="Cz"
    )
    assert electrodes == ["f3", "cz"]
    assert R.tolist() == [[1.0, -1.0]]
