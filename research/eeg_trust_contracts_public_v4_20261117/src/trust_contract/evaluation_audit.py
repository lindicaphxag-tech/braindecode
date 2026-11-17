"""Leakage checks over declared metadata; no EEG data or unseen-subject claims.

A non-overlap in incomplete pretraining-corpus records is UNKNOWN, never held out.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AuditRecord:
    dataset: str
    subject: str
    session: str
    site: str
    recording_id: str
    partition: Literal["train", "validation", "test"]
    # Explicit, optional stable identifiers; never inferred from local labels.
    global_subject_id: str | None = None
    origin_recording_id: str | None = None
    # Caller-declared pseudonymous identity of a physical acquisition site.
    global_site_id: str | None = None


@dataclass(frozen=True)
class CorpusStatement:
    datasets: frozenset[str]
    complete: bool
    source_reference: str | None


def audit_evaluation(records: list[AuditRecord], pretraining: CorpusStatement,
                     *, require_site_holdout: bool = False) -> dict:
    if not records:
        raise ValueError("evaluation records must be supplied")
    required = ["dataset", "subject", "session", "site", "recording_id"]
    for record in records:
        if any(not getattr(record, field) for field in required):
            raise ValueError("missing mandatory split metadata")
        for field in ("global_subject_id", "origin_recording_id", "global_site_id"):
            value = getattr(record, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"invalid {field}; use nonempty string or None")
        if record.partition not in ("train", "validation", "test"):
            raise ValueError("unknown evaluation partition")
    partitions = {part: [r for r in records if r.partition == part]
                  for part in ("train", "validation", "test")}
    violations = []
    for required_partition in ("train", "test"):
        if not partitions[required_partition]:
            violations.append({"kind": "missing_partition", "partition": required_partition})
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = partitions[left], partitions[right]
        subjects_a = {(r.dataset, r.subject) for r in a}
        subjects_b = {(r.dataset, r.subject) for r in b}
        overlap = sorted(subjects_a & subjects_b)
        if overlap:
            violations.append({"kind": "subject_overlap", "between": [left, right], "keys": overlap})
        rec_a = {(r.dataset, r.recording_id) for r in a}
        rec_b = {(r.dataset, r.recording_id) for r in b}
        if rec_a & rec_b:
            violations.append({"kind": "recording_overlap", "between": [left, right],
                               "keys": sorted(rec_a & rec_b)})
        global_a = {r.global_subject_id for r in a if r.global_subject_id is not None}
        global_b = {r.global_subject_id for r in b if r.global_subject_id is not None}
        if global_a & global_b:
            violations.append({"kind": "global_subject_overlap", "between": [left, right],
                               "keys": sorted(global_a & global_b)})
        origin_a = {r.origin_recording_id for r in a if r.origin_recording_id is not None}
        origin_b = {r.origin_recording_id for r in b if r.origin_recording_id is not None}
        if origin_a & origin_b:
            violations.append({"kind": "origin_recording_overlap", "between": [left, right],
                               "keys": sorted(origin_a & origin_b)})
        if require_site_holdout and a and b:
            sites = sorted({(r.dataset, r.site) for r in a} & {(r.dataset, r.site) for r in b})
            if sites:
                violations.append({"kind": "site_overlap", "between": [left, right], "keys": sites})
            stable_a = {r.global_site_id for r in a if r.global_site_id is not None}
            stable_b = {r.global_site_id for r in b if r.global_site_id is not None}
            if stable_a & stable_b:
                violations.append({"kind": "global_site_overlap", "between": [left, right],
                                   "keys": sorted(stable_a & stable_b)})
            # Different dataset-local site labels do not certify physical separation.
            if len({r.dataset for r in a + b}) > 1 and any(
                r.global_site_id is None for r in a + b
            ):
                violations.append({"kind": "site_holdout_unverified", "between": [left, right],
                                   "reason": "global_site_id missing in cross-dataset comparison"})
    evaluated = {r.dataset for r in partitions["validation"] + partitions["test"]}
    if not evaluated or pretraining.source_reference is None or not pretraining.datasets:
        corpus = "UNKNOWN"
    elif evaluated & pretraining.datasets:
        corpus = "IN_CORPUS_DECLARED"
    elif pretraining.complete:
        corpus = "NO_NAMED_OVERLAP_CANDIDATE"
    else:
        corpus = "UNKNOWN"
    identity_coverage = {
        part: {"records": len(subset),
               "global_subject_ids": sum(r.global_subject_id is not None for r in subset),
               "origin_recording_ids": sum(r.origin_recording_id is not None for r in subset),
               "global_site_ids": sum(r.global_site_id is not None for r in subset)}
        for part, subset in partitions.items()
    }
    fully_declared = all(r.global_subject_id is not None and
                         r.origin_recording_id is not None for r in records)
    return {"split_valid": not violations, "violations": violations,
            "cross_dataset_identity_status": "CHECKED_DECLARED_GLOBAL_IDS" if fully_declared else "PARTIAL_OR_UNAVAILABLE",
            "identity_coverage": identity_coverage,
            "site_holdout_status": (
                "NOT_REQUESTED" if not require_site_holdout else
                "BLOCKED_BY_UNKNOWN_SITE_OR_OVERLAP" if any(
                    v["kind"] in ("site_overlap", "global_site_overlap", "site_holdout_unverified")
                    for v in violations
                ) else "CHECKED_DECLARED_SITE_IDS"
            ),
            "corpus_assessment": corpus,
            "evaluated_datasets": sorted(evaluated),
            "declared_pretraining_overlap": sorted(evaluated & pretraining.datasets),
            "claim_boundary": "Metadata checks cannot certify global subject identity or absence from undocumented pretraining corpora; optional global identity and recording-origin keys are caller declarations and cannot verify their own truthfulness"}
