"""Synthetic audit matrix; does not load a real EEG checkpoint."""
from itertools import product
import json

import pytest
from trust_contract.validation_policy import Evidence, Decision, assess

P = frozenset({"backbone.weight", "adapter.weight", "head.weight"})
LOADED = frozenset({"backbone.weight"})


def make(**overrides):
    values = dict(workload="frozen_features", signal_status="EXACT",
                  checkpoint_parameters=P, loaded_parameters=LOADED,
                  feature_scope=frozenset({"backbone.weight"}), overlap="NONE",
                  claim_held_out=True)
    values.update(overrides)
    return Evidence(**values)


def test_valid_frozen_heldout_is_accepted():
    result = assess(make())
    assert result.decision == Decision.ACCEPT
    assert "declared_feature_parameters_loaded" in result.guarantees
    assert json.loads(result.as_json())["decision"] == "ACCEPT"


def test_unknown_signal_never_accepted():
    r = assess(make(signal_status="UNKNOWN"))
    assert r.decision == Decision.REVIEW
    assert "signal_metadata_unverified" in r.reasons


def test_unreconstructible_signal_refused():
    assert assess(make(signal_status="INCOMPATIBLE")).decision == Decision.REFUSE


def test_transform_requires_executed_verified_mapping():
    assert assess(make(signal_status="TRANSFORMABLE")).decision == Decision.REVIEW
    assert assess(make(signal_status="TRANSFORMABLE", channel_transform_applied=True)).decision == Decision.ACCEPT


def test_unloaded_frozen_adapter_refused():
    r = assess(make(feature_scope=frozenset({"backbone.weight", "adapter.weight"})))
    assert r.decision == Decision.REFUSE
    assert r.missing_feature_parameters == ("adapter.weight",)


@pytest.mark.parametrize("field", ["checkpoint_parameters", "loaded_parameters", "feature_scope"])
def test_missing_frozen_provenance_requires_review(field):
    assert assess(make(**{field: None})).decision == Decision.REVIEW


def test_empty_feature_scope_is_not_a_false_pass():
    assert assess(make(feature_scope=frozenset())).decision == Decision.REVIEW


def test_finetuning_allows_new_head_but_discloses_missing_provenance():
    e = make(workload="fine_tune", loaded_parameters=None, feature_scope=None,
             checkpoint_parameters=None, claim_held_out=False, overlap="NONE")
    assert assess(e).decision == Decision.DISCLOSE


def test_in_corpus_result_is_labeled_not_called_heldout():
    assert assess(make(claim_held_out=False, overlap="OVERLAP")).decision == Decision.DISCLOSE
    assert assess(make(claim_held_out=True, overlap="OVERLAP")).decision == Decision.REFUSE


def test_heldout_unknown_is_review_not_accept():
    assert assess(make(overlap="UNKNOWN")).decision == Decision.REVIEW


def test_invalid_loaded_or_scope_rejected():
    with pytest.raises(ValueError, match="actual model parameters"):
        make(loaded_parameters=frozenset({"nonexistent"}))
    with pytest.raises(ValueError, match="nonexistent model parameter"):
        make(feature_scope=frozenset({"nonexistent"}))


@pytest.mark.parametrize("field,value", [("signal_status", "signed"), ("workload", "probe"),
                                         ("overlap", "maybe")])
def test_invalid_vocab_rejected(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


def test_model_parameter_sets_have_strict_type():
    with pytest.raises(TypeError, match="loaded_parameters"):
        make(loaded_parameters={"backbone.weight"})


@pytest.mark.parametrize("signal,overlap", product(
    ["EXACT", "TRANSFORMABLE", "INCOMPATIBLE", "UNKNOWN"],
    ["NONE", "OVERLAP", "UNKNOWN"]))
def test_heldout_cannot_accept_unknown_or_overlap(signal, overlap):
    report = assess(make(signal_status=signal, overlap=overlap))
    if overlap != "NONE" or signal != "EXACT":
        assert report.decision != Decision.ACCEPT


def test_determinism_and_sorted_missing_parameters():
    missing = frozenset({"head.weight", "adapter.weight"})
    a = assess(make(feature_scope=frozenset({"backbone.weight"}) | missing))
    b = assess(make(feature_scope=frozenset({"backbone.weight"}) | missing))
    assert a.as_json() == b.as_json()
    assert a.missing_feature_parameters == ("adapter.weight", "head.weight")


def test_finetuning_zero_loaded_weights_cannot_be_accepted():
    # Known empty checkpoint coverage must not be mistaken for transfer.
    report = assess(make(
        workload="fine_tune",
        feature_scope=None,
        loaded_parameters=frozenset(),
    ))
    assert report.decision == Decision.REFUSE
    assert "zero_pretrained_parameters_loaded" in report.reasons
