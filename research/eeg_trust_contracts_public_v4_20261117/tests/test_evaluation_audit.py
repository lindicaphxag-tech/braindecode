"""Metadata leakage checks on synthetic records only."""
import pytest
from trust_contract.evaluation_audit import AuditRecord, CorpusStatement, audit_evaluation


def rec(subject, partition, *, dataset="D", site="siteA", session="session1", recording=None):
    return AuditRecord(dataset, subject, session, site, recording or f"rec_{subject}_{session}", partition)


def corpus(*datasets, complete=False, source="paper:appendix"):
    return CorpusStatement(frozenset(datasets), complete, source)


def test_subject_overlap_even_across_sessions():
    x = audit_evaluation([rec("S1", "train"), rec("S1", "test", session="session2")], corpus("D2"))
    assert not x["split_valid"]
    assert any(v["kind"] == "subject_overlap" for v in x["violations"])


def test_recording_overlap_across_subject_ids():
    x = audit_evaluation([rec("S1", "train", recording="R"), rec("S2", "test", recording="R")], corpus("D2"))
    assert any(v["kind"] == "recording_overlap" for v in x["violations"])


def test_site_holdout_opt_in():
    rows = [rec("S1", "train"), rec("S2", "test")]
    assert audit_evaluation(rows, corpus("D2"))["split_valid"]
    assert not audit_evaluation(rows, corpus("D2"), require_site_holdout=True)["split_valid"]


def test_site_independence():
    assert audit_evaluation([rec("S1", "train", site="A"), rec("S2", "test", site="B")],
                            corpus("D2"), require_site_holdout=True)["split_valid"]


def test_declared_pretraining_overlap():
    x = audit_evaluation([rec("S2", "test")], corpus("D"))
    assert x["corpus_assessment"] == "IN_CORPUS_DECLARED"


def test_incomplete_pretraining_is_unknown():
    x = audit_evaluation([rec("S2", "test")], corpus("D2", complete=False))
    assert x["corpus_assessment"] == "UNKNOWN"


def test_complete_nonoverlap_is_only_candidate():
    x = audit_evaluation([rec("S2", "test")], corpus("D2", complete=True))
    assert x["corpus_assessment"] == "NO_NAMED_OVERLAP_CANDIDATE"


def test_absent_pretraining_source_unknown():
    assert audit_evaluation([rec("S2", "test")], corpus("D2", source=None))["corpus_assessment"] == "UNKNOWN"


def test_cross_dataset_subject_alias_cannot_be_certified():
    x = audit_evaluation([rec("001", "train", dataset="D1"), rec("001", "test", dataset="D2")], corpus("D3"))
    assert x["split_valid"]
    assert "cannot certify" in x["claim_boundary"]


def test_missing_site_rejected():
    with pytest.raises(ValueError, match="mandatory"):
        audit_evaluation([rec("S1", "test", site="")], corpus("D2"))


def test_empty_rejected():
    with pytest.raises(ValueError, match="supplied"):
        audit_evaluation([], corpus("D2"))

def test_missing_train_partition_is_invalid():
    x = audit_evaluation([rec("S2", "test")], corpus("D2"))
    assert not x["split_valid"]
    assert any(v["kind"] == "missing_partition" and v["partition"] == "train" for v in x["violations"])


def test_missing_test_partition_is_invalid_and_not_heldout_candidate():
    x = audit_evaluation([rec("S1", "train")], corpus("D2", complete=True))
    assert not x["split_valid"] and x["corpus_assessment"] == "UNKNOWN"


def test_unknown_partition_rejected():
    with pytest.raises(ValueError, match="unknown evaluation partition"):
        audit_evaluation([rec("S1", "train"), rec("S2", "untracked")], corpus("D2"))


def test_cross_dataset_same_global_subject_is_detected():
    a = AuditRecord("dataset_A", "S01", "ses1", "site_A", "a1", "train", "gid_001")
    b = AuditRecord("dataset_B", "subject-abc", "ses2", "site_B", "b9", "test", "gid_001")
    result = audit_evaluation([a, b], corpus("other"))
    assert not result["split_valid"]
    assert any(v["kind"] == "global_subject_overlap" for v in result["violations"])


def test_cross_dataset_same_original_recording_is_detected():
    a = AuditRecord("dataset_A", "A", "ses1", "site_A", "a1", "train", None, "orig_001")
    b = AuditRecord("dataset_B", "B", "ses2", "site_B", "b9", "test", None, "orig_001")
    result = audit_evaluation([a, b], corpus("other"))
    assert not result["split_valid"]
    assert any(v["kind"] == "origin_recording_overlap" for v in result["violations"])


def test_declared_global_keys_are_not_independently_verified():
    a = AuditRecord("dataset_A", "S01", "ses1", "site_A", "a1", "train", "gid_1", "orig_1")
    b = AuditRecord("dataset_B", "S01", "ses1", "site_B", "b9", "test", "gid_2", "orig_2")
    result = audit_evaluation([a, b], corpus("other"))
    assert result["split_valid"]
    assert result["cross_dataset_identity_status"] == "CHECKED_DECLARED_GLOBAL_IDS"
    assert "truthfulness" in result["claim_boundary"]


@pytest.mark.parametrize("invalid_key", ["", "   ", 123])
def test_invalid_global_subject_id_is_rejected(invalid_key):
    a = AuditRecord("A", "S1", "s", "site", "one", "train", invalid_key)
    with pytest.raises(ValueError, match="global_subject_id"):
        audit_evaluation([a, rec("S2", "test")], corpus("other"))


def test_invalid_origin_recording_id_is_rejected():
    a = AuditRecord("A", "S1", "s", "site", "one", "train", None, "")
    with pytest.raises(ValueError, match="origin_recording_id"):
        audit_evaluation([a, rec("S2", "test")], corpus("other"))


def test_partial_identity_coverage_is_explicit():
    a = AuditRecord("A", "S1", "s", "site", "one", "train", "gid1", "rid1")
    b = AuditRecord("B", "S2", "s", "site", "two", "test")
    result = audit_evaluation([a, b], corpus("other"))
    assert result["split_valid"]
    assert result["cross_dataset_identity_status"] == "PARTIAL_OR_UNAVAILABLE"
    assert result["identity_coverage"]["test"]["global_subject_ids"] == 0


def test_cross_dataset_identity_is_checked_for_validation_overlap():
    a = AuditRecord("A", "S1", "s", "site", "one", "train", "gid1")
    b = AuditRecord("B", "S2", "s", "site", "two", "validation", "gid1")
    c = AuditRecord("C", "S3", "s", "site", "three", "test", "gid3")
    result = audit_evaluation([a, b, c], corpus("other"))
    assert any(v["kind"] == "global_subject_overlap" and v["between"] == ["train", "validation"] for v in result["violations"])

def test_cross_dataset_missing_global_site_blocks_strict_holdout():
    a = AuditRecord("D1", "S1", "ses1", "Hospital X", "r1", "train")
    b = AuditRecord("D2", "S2", "ses2", "Clinic Y", "r2", "test")
    assert audit_evaluation([a, b], corpus("other"))["split_valid"]
    strict = audit_evaluation([a, b], corpus("other"), require_site_holdout=True)
    assert not strict["split_valid"]
    assert strict["site_holdout_status"] == "BLOCKED_BY_UNKNOWN_SITE_OR_OVERLAP"
    assert any(v["kind"] == "site_holdout_unverified" for v in strict["violations"])


def test_cross_dataset_physical_site_alias_detected():
    a = AuditRecord("D1", "S1", "ses1", "Hospital X", "r1", "train",
                    global_site_id="physical-1")
    b = AuditRecord("D2", "S2", "ses2", "Clinic Y", "r2", "test",
                    global_site_id="physical-1")
    result = audit_evaluation([a, b], corpus("other"), require_site_holdout=True)
    assert not result["split_valid"]
    assert any(v["kind"] == "global_site_overlap" for v in result["violations"])


def test_distinct_declared_physical_sites_pass_only_metadata_preflight():
    a = AuditRecord("D1", "S1", "ses1", "Hospital", "r1", "train",
                    global_site_id="physical-1")
    b = AuditRecord("D2", "S2", "ses2", "Hospital", "r2", "test",
                    global_site_id="physical-2")
    result = audit_evaluation([a, b], corpus("other"), require_site_holdout=True)
    assert result["split_valid"]
    assert result["site_holdout_status"] == "CHECKED_DECLARED_SITE_IDS"
    assert result["identity_coverage"]["test"]["global_site_ids"] == 1


def test_partial_global_site_ids_report_unverified():
    a = AuditRecord("D1", "S1", "s", "A", "r1", "train",
                    global_site_id="physical-1")
    b = AuditRecord("D2", "S2", "s", "B", "r2", "test")
    result = audit_evaluation([a, b], corpus("other"), require_site_holdout=True)
    assert any(v["kind"] == "site_holdout_unverified" for v in result["violations"])


@pytest.mark.parametrize("site_id", ["", "   ", 42])
def test_invalid_global_site_id_rejected(site_id):
    a = AuditRecord("D1", "S1", "s", "siteA", "r1", "train",
                    global_site_id=site_id)
    with pytest.raises(ValueError, match="global_site_id"):
        audit_evaluation([a, rec("S2", "test")], corpus("other"))


def test_same_dataset_local_site_holdout_unchanged():
    result = audit_evaluation(
        [rec("S1", "train", site="A"), rec("S2", "test", site="B")],
        corpus("other"), require_site_holdout=True,
    )
    assert result["split_valid"]
    assert result["site_holdout_status"] == "CHECKED_DECLARED_SITE_IDS"
