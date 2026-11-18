"""Integration between provenance metadata and independent checkpoint bytes."""
import copy
import torch

from trust_contract.model_bridge import inspect_model
from trust_contract.validation_policy import Decision


def make():
    model = torch.nn.Module()
    model.encoder = torch.nn.BatchNorm1d(3)
    model.__dict__["_pretrained_provenance"] = {
        "repo_id": "example/eeg-model",
        "commit": "a" * 40,
        "loaded": set(model.state_dict()),
        "mixed": set(),
    }
    return model


def check(model, state, feature_scope):
    return inspect_model(
        model, signal_status="EXACT",
        feature_scope=feature_scope, overlap="NONE",
        claim_held_out=True, checkpoint_reference_state=state,
    )


def test_real_checkpoint_snapshot_can_attest_batchnorm_and_weights():
    model = make()
    reference = copy.deepcopy(model.state_dict())
    out = check(model, reference, feature_scope={
        "encoder.weight", "encoder.running_var",
    })
    assert out.report.decision == Decision.ACCEPT
    assert out.tensor_attestation.verified


def test_reference_mismatch_refuses_frozen_features():
    model = make()
    reference = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.encoder.running_var.add_(1.0)
    out = check(model, reference, feature_scope={
        "encoder.weight", "encoder.running_var",
    })
    assert out.report.decision == Decision.REFUSE
    assert out.tensor_attestation.different == ("encoder.running_var",)


def test_live_self_reference_refuses_frozen_feature_claim():
    model = make()
    reference = model.state_dict()  # intentionally aliased
    out = check(model, reference, feature_scope={
        "encoder.weight", "encoder.running_var",
    })
    assert out.report.decision == Decision.REFUSE
    assert not out.tensor_attestation.verified


def test_partial_finetune_reference_does_not_silently_accept():
    model = make()
    reference = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.encoder.bias.add_(2.0)
    out = inspect_model(
        model, signal_status="EXACT", workload="fine_tune",
        overlap="NONE", claim_held_out=True,
        checkpoint_reference_state=reference,
    )
    assert out.report.decision == Decision.REVIEW
    assert "checkpoint_reference_not_fully_verified" in out.report.reasons
