"""Offline, read-only byte-level comparison of selected PyTorch checkpoint state.

This verifies model tensors against the caller-supplied reference state, not
where the reference came from. Metadata pins or model cards are not evidence of
artifact authenticity. Unverifiable sparse, meta or uninitialized tensors fail
closed; model buffers must be explicitly included in required_keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import hashlib


@dataclass(frozen=True)
class StateAttestation:
    matched: tuple[str, ...]
    missing_in_model: tuple[str, ...]
    missing_in_reference: tuple[str, ...]
    different: tuple[str, ...]
    unverifiable: tuple[str, ...]
    comparison_sha256: str

    @property
    def verified(self) -> bool:
        return bool(self.matched) and not any((
            self.missing_in_model, self.missing_in_reference,
            self.different, self.unverifiable,
        ))


def _bytes(tensor: Any) -> bytes:
    """Serialize raw tensor storage as bytes without numeric conversions.

    Rejects non-dense / meta / lazy tensors. Uses CPU *copies* for comparison,
    leaving source data and gradients unchanged. Dtype/shape checked separately.
    """
    import torch
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a dense torch.Tensor")
    from torch.nn.parameter import is_lazy
    if is_lazy(tensor) or tensor.device.type == "meta" or tensor.layout != torch.strided:
        raise ValueError("tensor is unmaterialized or has unsupported storage")
    arr = tensor.detach().to(device="cpu", copy=True).contiguous()
    # Byte-view is defined for bool, integer, fp16, bf16, fp32, fp64 and other
    # dense dtypes and retains distinctions like +0/-0 and NaN payload bits.
    # Scalar state (e.g. BatchNorm.num_batches_tracked) must gain a
    # dimension before reinterpreting multi-byte dtypes as uint8.
    return arr.reshape(-1).view(torch.uint8).numpy().tobytes()


def attest_checkpoint_state(
    model: Any,
    reference_state: Mapping[str, Any],
    *,
    required_keys: Iterable[str],
) -> StateAttestation:
    """Compare declared model state keys to exact caller-provided reference bytes.

    Required keys must be independently derived from the executed inference
    path. To assert *all* persistent state, pass model.state_dict().keys().
    This comparator does not determine provenance, data-set leakage, feature
    reachability, or whether a reference was signed or downloaded correctly.
    """
    if isinstance(required_keys, (str, bytes)):
        raise TypeError("required_keys must be an iterable of complete names")
    keys = tuple(required_keys)
    if not keys or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("required_keys must contain nonempty state names")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate required state names")
    observed = model.state_dict()
    matched, no_model, no_reference, different, unverifiable = [], [], [], [], []
    digest = hashlib.sha256()
    for name in sorted(keys):
        if name not in observed:
            no_model.append(name)
            continue
        if name not in reference_state:
            no_reference.append(name)
            continue
        actual, expected = observed[name], reference_state[name]
        try:
            import torch
            # model.state_dict() returns *views* of live model storage. Passing
            # it back as a "reference" would silently create a tautological
            # self-comparison: subsequent training mutates both sides. Reject
            # aliased storage and demand an independently snapshotted state.
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                if (actual.layout == torch.strided and expected.layout == torch.strided
                        and actual.device == expected.device
                        and actual.numel() > 0 and expected.numel() > 0
                        and actual.untyped_storage().data_ptr()
                        == expected.untyped_storage().data_ptr()):
                    unverifiable.append(name)
                    continue
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                different.append(name)
                continue
            got = _bytes(actual)
            want = _bytes(expected)
        except (AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError):
            unverifiable.append(name)
            continue
        if got != want:
            different.append(name)
            continue
        matched.append(name)
        # Include ordered logical identities, dtype and dimensions to
        # prevent concatenation ambiguity between unrelated state layouts.
        tag = f"{name}|{actual.dtype}|{list(actual.shape)}|{len(got)}".encode()
        digest.update(len(tag).to_bytes(8, "big"))
        digest.update(tag)
        digest.update(len(got).to_bytes(8, "big"))
        digest.update(got)
    return StateAttestation(
        tuple(matched), tuple(no_model), tuple(no_reference),
        tuple(different), tuple(unverifiable), digest.hexdigest(),
    )
