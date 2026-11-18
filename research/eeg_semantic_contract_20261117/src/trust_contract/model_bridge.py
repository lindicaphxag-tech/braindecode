"""Read-only adapter for Braindecode's existing checkpoint provenance.

This is a standalone research helper, not an upstream model API. It cannot
infer the executed feature graph, validate physical electrodes, or establish
dataset disjointness. It never downloads weights or mutates the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re

from .validation_policy import Evidence, Report, assess

_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


@dataclass(frozen=True)
class ModelAudit:
    evidence: Evidence
    report: Report
    checkpoint_pinned: bool
    card_datasets: tuple[str, ...]
    observed_parameter_names: tuple[str, ...]


def inspect_model(
    model: Any,
    *,
    signal_status: str,
    feature_scope: Iterable[str] | None = None,
    workload: str = "frozen_features",
    channel_transform_applied: bool = False,
    overlap: str = "UNKNOWN",
    claim_held_out: bool = False,
) -> ModelAudit:
    """Inspect current parameter identities and recorded checkpoint lineage.

    ``feature_scope`` MUST be independently established from an actual execution
    trace or a manually reviewed graph. Providing it does not validate that
    trace; leaving it unset makes frozen feature extraction require REVIEW.
    ``overlap`` MUST come from an independently validated corpus-identity check.
    Absence of a model card never implies disjointness.

    Braindecode's ``from_pretrained`` stores ``_pretrained_provenance`` with a
    ``loaded`` set and an optional ``mixed`` set. Only an immutable 40-hex
    checkpoint ``commit`` is considered pinned here, and only current
    parameter objects are included in a loaded-feature assurance.
    """
    # Inference state includes BatchNorm-like running buffers, not just
    # trainable parameters. A fully loaded parameter set alone is not proof
    # of a fully restored frozen feature path.
    pairs = list(model.named_parameters(remove_duplicate=False))
    if callable(getattr(model, "named_buffers", None)):
        pairs.extend(model.named_buffers(remove_duplicate=False))
    tensors = dict(pairs)
    names = frozenset(tensors)
    if isinstance(feature_scope, (str, bytes)):
        raise TypeError("feature_scope must contain full parameter names, not a string")
    scope = frozenset(feature_scope) if feature_scope is not None else None

    prov = model.__dict__.get("_pretrained_provenance")
    prov = prov if isinstance(prov, dict) else {}
    pinned = (
        isinstance(prov.get("repo_id"), str)
        and bool(prov["repo_id"].strip())
        and isinstance(prov.get("commit"), str)
        and bool(_COMMIT.fullmatch(prov["commit"]))
    )
    # state_dict() enumerates checkpoint-persistent state. A nonpersistent
    # buffer may affect outputs but cannot be attributed to saved weights.
    persistent = (
        set(model.state_dict())
        if callable(getattr(model, "state_dict", None))
        else set(names)
    )
    loaded: frozenset[str] | None = None
    if pinned and isinstance(prov.get("loaded"), (set, frozenset, list, tuple)):
        loaded_ids = {
            id(tensors[name]) for name in set(prov["loaded"]) & names & persistent
        }
        mixed_ids = {
            id(tensors[name]) for name in set(prov.get("mixed") or ()) & names
        }
        try:
            from torch.nn.parameter import is_lazy
        except ImportError:
            is_lazy = lambda _: False
        loaded = frozenset(
            name for name, tensor in pairs
            if name in persistent
            and id(tensor) in loaded_ids
            and id(tensor) not in mixed_ids
            and not is_lazy(tensor)
        )

    card = getattr(model, "card_data", None)
    datasets = getattr(card, "datasets", None) if card is not None else None
    if isinstance(datasets, str):
        datasets = [datasets]
    card_datasets = tuple(str(s) for s in datasets) if datasets else ()

    e = Evidence(
        workload=workload,
        signal_status=signal_status,
        channel_transform_applied=channel_transform_applied,
        checkpoint_parameters=names,
        loaded_parameters=loaded,
        feature_scope=scope,
        overlap=overlap,
        claim_held_out=claim_held_out,
    )
    return ModelAudit(
        evidence=e,
        report=assess(e),
        checkpoint_pinned=pinned,
        card_datasets=card_datasets,
        observed_parameter_names=tuple(sorted(names)),
    )
