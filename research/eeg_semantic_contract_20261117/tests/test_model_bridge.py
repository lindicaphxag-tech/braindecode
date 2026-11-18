"""No downloads, installed Braindecode, Torch or real checkpoint needed."""
from types import SimpleNamespace

import pytest
from trust_contract.model_bridge import inspect_model
from trust_contract.validation_policy import Decision

SHA = "a" * 40


class FakeModel:
    def __init__(self, *, loaded=True, commit=SHA, tied=False, mixed=()):
        self.encoder = object()
        self.head = object() if not tied else self.encoder
        self._parameters = [("encoder.weight", self.encoder), ("head.weight", self.head)]
        if loaded is not None:
            self._pretrained_provenance = {
                "commit": commit,
                "repo_id": "example/eeg-model",
                "loaded": {"encoder.weight"} if loaded else set(),
                "mixed": set(mixed),
            }
        self.card_data = SimpleNamespace(datasets=["ExampleEEG"])

    def named_parameters(self, *, remove_duplicate):
        assert remove_duplicate is False
        return iter(self._parameters)


def audit(model=None, **kw):
    kw = {"signal_status": "EXACT", "feature_scope": {"encoder.weight"},
          "overlap": "NONE", "claim_held_out": True, **kw}
    return inspect_model(model or FakeModel(), **kw)


def test_pinned_loaded_feature_is_accepted_under_supplied_evidence():
    result = audit()
    assert result.report.decision == Decision.ACCEPT
    assert result.checkpoint_pinned
    assert result.card_datasets == ("ExampleEEG",)


def test_tied_parameter_alias_matches_loaded_object():
    result = audit(FakeModel(tied=True), feature_scope={"head.weight"})
    assert result.evidence.loaded_parameters == frozenset({"encoder.weight", "head.weight"})
    assert result.report.decision == Decision.ACCEPT


def test_mixed_adapter_excludes_all_tied_aliases():
    result = audit(FakeModel(tied=True, mixed={"head.weight"}))
    assert result.report.decision == Decision.REFUSE
    assert result.report.missing_feature_parameters == ("encoder.weight",)


@pytest.mark.parametrize("revision", [None, "main", "abc123", "z" * 40])
def test_nonimmutable_revision_forces_review(revision):
    result = audit(FakeModel(commit=revision))
    assert result.report.decision == Decision.REVIEW
    assert result.evidence.loaded_parameters is None


def test_no_pretrained_provenance_forces_review():
    assert audit(FakeModel(loaded=None)).report.decision == Decision.REVIEW


def test_unloaded_feature_is_refused():
    result = audit(FakeModel(loaded=False))
    assert result.report.decision == Decision.REFUSE


def test_unknown_feature_scope_requires_review():
    result = inspect_model(FakeModel(), signal_status="EXACT", overlap="NONE", claim_held_out=True)
    assert result.report.decision == Decision.REVIEW


def test_model_card_datasets_do_not_automatically_claim_disjointness():
    result = inspect_model(FakeModel(), signal_status="EXACT",
                           feature_scope={"encoder.weight"}, claim_held_out=True)
    assert result.card_datasets
    assert result.report.decision == Decision.REVIEW


def test_no_mutation_during_inspection():
    model = FakeModel()
    old = (id(model.encoder), id(model.head), dict(model._pretrained_provenance))
    audit(model)
    assert old == (id(model.encoder), id(model.head), model._pretrained_provenance)


def test_declared_transform_requires_verified_application():
    result = inspect_model(FakeModel(), signal_status="TRANSFORMABLE",
                           feature_scope={"encoder.weight"}, overlap="NONE", claim_held_out=True)
    assert result.report.decision == Decision.REVIEW


def test_real_torch_parameter_alias_identity_if_available():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2, bias=False)
    model.head = torch.nn.Module()
    model.head.register_parameter("weight", model.encoder.weight)
    model.__dict__["_pretrained_provenance"] = {
        "commit": SHA, "repo_id": "example/eeg-model",
        "loaded": {"encoder.weight"},
        "mixed": set(),
    }
    report = inspect_model(
        model, signal_status="EXACT", feature_scope={"head.weight"},
        overlap="NONE", claim_held_out=True,
    )
    assert report.report.decision == Decision.ACCEPT
    assert report.evidence.loaded_parameters == frozenset({"encoder.weight", "head.weight"})


def test_lazy_unmaterialized_parameter_is_not_certified_as_pretrained():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.encoder = torch.nn.LazyLinear(3)
    model.__dict__["_pretrained_provenance"] = {
        "commit": SHA, "repo_id": "example/eeg-model", "loaded": {"encoder.weight"}, "mixed": set()
    }
    a = inspect_model(
        model, signal_status="EXACT", feature_scope={"encoder.weight"},
        overlap="NONE", claim_held_out=True,
    )
    assert a.report.decision == Decision.REFUSE
    assert a.report.missing_feature_parameters == ("encoder.weight",)


def test_string_feature_scope_is_rejected_instead_of_split_into_characters():
    with pytest.raises(TypeError, match="feature_scope"):
        inspect_model(
            FakeModel(), signal_status="EXACT", feature_scope="encoder.weight"
        )

def test_hash_without_checkpoint_repository_identity_requires_review():
    model = FakeModel()
    model._pretrained_provenance.pop("repo_id")
    report = audit(model)
    assert not report.checkpoint_pinned
    assert report.report.decision == Decision.REVIEW

def test_batchnorm_unloaded_running_statistics_fail_frozen_feature_gate():
    """Checkpointed weights do not imply checkpointed inference statistics."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.encoder = torch.nn.BatchNorm1d(3)
    model.__dict__["_pretrained_provenance"] = {
        "repo_id": "example/eeg-model",
        "commit": SHA,
        "loaded": {"encoder.weight", "encoder.bias"},
        "mixed": set(),
    }
    report = inspect_model(
        model, signal_status="EXACT", overlap="NONE", claim_held_out=True,
        feature_scope={
            "encoder.weight", "encoder.bias",
            "encoder.running_mean", "encoder.running_var",
        },
    )
    assert report.report.decision == Decision.REFUSE
    assert report.report.missing_feature_parameters == (
        "encoder.running_mean", "encoder.running_var"
    )


def test_batchnorm_checkpointed_running_statistics_can_pass_declared_scope():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.encoder = torch.nn.BatchNorm1d(3)
    model.__dict__["_pretrained_provenance"] = {
        "repo_id": "example/eeg-model",
        "commit": SHA,
        "loaded": set(model.state_dict()),
        "mixed": set(),
    }
    report = inspect_model(
        model, signal_status="EXACT", overlap="NONE", claim_held_out=True,
        feature_scope={
            "encoder.weight", "encoder.bias",
            "encoder.running_mean", "encoder.running_var",
        },
    )
    assert report.report.decision == Decision.ACCEPT


def test_nonpersistent_buffer_cannot_be_attributed_to_checkpoint():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.register_buffer("nonpersistent_scale", torch.ones(3), persistent=False)
    model.__dict__["_pretrained_provenance"] = {
        "repo_id": "example/eeg-model",
        "commit": SHA,
        "loaded": {"nonpersistent_scale"},
        "mixed": set(),
    }
    report = inspect_model(
        model, signal_status="EXACT", overlap="NONE", claim_held_out=True,
        feature_scope={"nonpersistent_scale"},
    )
    assert report.report.decision == Decision.REFUSE
    assert report.report.missing_feature_parameters == ("nonpersistent_scale",)
