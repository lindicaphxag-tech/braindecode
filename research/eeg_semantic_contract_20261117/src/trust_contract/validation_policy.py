"""External, opt-in decision policy for EEG foundation-model evidence.

Consumes existing model-card, channel-compatibility and load provenance results;
creates no new Braindecode loading, inference or checkpoint APIs.

The gate reports evidence sufficiency, not accuracy, scientific validity of
unverified metadata, or a certification of clinical safety.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import FrozenSet, Literal


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    DISCLOSE = "DISCLOSE"
    REVIEW = "REVIEW"
    REFUSE = "REFUSE"


SignalStatus = Literal["EXACT", "TRANSFORMABLE", "INCOMPATIBLE", "UNKNOWN"]
OverlapStatus = Literal["NONE", "OVERLAP", "UNKNOWN"]
Workload = Literal["frozen_features", "fine_tune"]


@dataclass(frozen=True)
class Evidence:
    """Typed, declarative input. An empty declaration is not a passed gate.

    channel_transform_applied: whether an explicit mapping was applied and
      tested on this run, not merely mathematically available.
    checkpoint_parameters: names of parameters observable in the actual model.
    loaded_parameters: verified names loaded from the pinned checkpoint, reused
      from existing Braindecode _pretrained_provenance, when available.
    feature_scope: explicitly declared parameters of the inspected feature path;
      must be supplied externally; this gate does NOT infer the forward graph.
    overlap: status returned by an independently justified pretraining-dataset
      overlap check. If no model-card evidence is present it must be UNKNOWN.
    """
    workload: Workload
    signal_status: SignalStatus
    channel_transform_applied: bool = False
    checkpoint_parameters: FrozenSet[str] | None = None
    loaded_parameters: FrozenSet[str] | None = None
    feature_scope: FrozenSet[str] | None = None
    overlap: OverlapStatus = "UNKNOWN"
    claim_held_out: bool = False

    def __post_init__(self) -> None:
        if self.workload not in {"frozen_features", "fine_tune"}:
            raise ValueError("unsupported workload")
        if self.signal_status not in {"EXACT", "TRANSFORMABLE", "INCOMPATIBLE", "UNKNOWN"}:
            raise ValueError("unsupported signal status")
        if self.overlap not in {"NONE", "OVERLAP", "UNKNOWN"}:
            raise ValueError("unsupported overlap status")
        for field in ("checkpoint_parameters", "loaded_parameters", "feature_scope"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, frozenset):
                raise TypeError(f"{field} must be frozenset or None")
        if self.checkpoint_parameters is not None and self.loaded_parameters is not None:
            if not self.loaded_parameters <= self.checkpoint_parameters:
                raise ValueError("loaded parameters must be actual model parameters")
        if self.checkpoint_parameters is not None and self.feature_scope is not None:
            if not self.feature_scope <= self.checkpoint_parameters:
                raise ValueError("feature scope includes a nonexistent model parameter")


@dataclass(frozen=True)
class Report:
    decision: Decision
    reasons: tuple[str, ...]
    missing_feature_parameters: tuple[str, ...]
    guarantees: tuple[str, ...]

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


def assess(e: Evidence) -> Report:
    """Fail closed on known incompatibility; review unknown frozen evidence.

    ACCEPT only says the *provided evidence* meets this narrow policy. It does
    not verify the physical acquisition manifest or generalization performance.
    """
    reasons: list[str] = []
    missing: tuple[str, ...] = ()
    guarantees: list[str] = []
    severity = Decision.ACCEPT
    order = {Decision.ACCEPT: 0, Decision.DISCLOSE: 1,
             Decision.REVIEW: 2, Decision.REFUSE: 3}

    def mark(decision: Decision, reason: str) -> None:
        nonlocal severity
        if order[decision] > order[severity]:
            severity = decision
        reasons.append(reason)

    if e.signal_status == "INCOMPATIBLE":
        mark(Decision.REFUSE, "signal_derivation_incompatible")
    elif e.signal_status == "UNKNOWN":
        mark(Decision.REVIEW, "signal_metadata_unverified")
    elif e.signal_status == "TRANSFORMABLE":
        if not e.channel_transform_applied:
            mark(Decision.REVIEW, "channel_transform_not_applied_and_verified")
        else:
            guarantees.append("declared_channel_linear_transform_applied")
    else:
        guarantees.append("declared_signal_identity")

    if e.workload == "frozen_features":
        if (e.checkpoint_parameters is None or e.loaded_parameters is None
                or e.feature_scope is None or not e.feature_scope):
            mark(Decision.REVIEW, "frozen_feature_parameter_scope_unverified")
        else:
            missing = tuple(sorted(e.feature_scope - e.loaded_parameters))
            if missing:
                mark(Decision.REFUSE, "frozen_feature_path_contains_unloaded_parameters")
            else:
                guarantees.append("declared_feature_parameters_loaded")
    else:
        if e.loaded_parameters is None:
            mark(Decision.DISCLOSE, "checkpoint_loading_unverified_for_finetuning")
        elif not e.loaded_parameters:
            # A model with zero loaded checkpoint tensors is not evidence of
            # transfer learning, even if model-card metadata looks complete.
            mark(Decision.REFUSE, "zero_pretrained_parameters_loaded")

    if e.claim_held_out:
        if e.overlap == "OVERLAP":
            mark(Decision.REFUSE, "pretraining_corpus_overlaps_claimed_heldout_evaluation")
        elif e.overlap == "UNKNOWN":
            mark(Decision.REVIEW, "heldout_claim_lacks_pretraining_corpus_evidence")
        else:
            guarantees.append("declared_pretraining_corpus_disjoint")
    elif e.overlap == "OVERLAP":
        mark(Decision.DISCLOSE, "in_corpus_evaluation_must_be_labeled")
    elif e.overlap == "UNKNOWN":
        mark(Decision.DISCLOSE, "pretraining_corpus_unverified")

    if not reasons:
        reasons.append("supplied_evidence_satisfies_selected_policy")
    return Report(severity, tuple(reasons), missing, tuple(guarantees))
