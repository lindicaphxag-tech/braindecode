import copy
import pytest
import torch
from trust_contract.state_attestation import attest_checkpoint_state as attest


def linear():
    model = torch.nn.Linear(3, 2)
    return model, copy.deepcopy(model.state_dict())


def test_unmodified_linear_matches_all_state_bytes():
    model, ref = linear()
    result = attest(model, ref, required_keys=ref.keys())
    assert result.verified and len(result.matched) == len(ref)


def test_weight_mutation_detected_without_mutating_reference():
    model, ref = linear()
    old = ref["weight"].clone()
    with torch.no_grad():
        model.weight[0, 0] += 1
    out = attest(model, ref, required_keys=ref.keys())
    assert not out.verified and out.different == ("weight",)
    assert torch.equal(ref["weight"], old)


def test_missing_reference_fails_closed():
    model, ref = linear()
    del ref["bias"]
    out = attest(model, ref, required_keys=("weight", "bias"))
    assert out.missing_in_reference == ("bias",) and not out.verified


def test_missing_model_state_fails_closed():
    model, ref = linear()
    out = attest(model, ref, required_keys=("weight", "not_in_model"))
    assert out.missing_in_model == ("not_in_model",)


def test_shape_and_dtype_must_match_without_cast():
    model, ref = linear()
    ref["weight"] = ref["weight"].double()
    out = attest(model, ref, required_keys=("weight",))
    assert out.different == ("weight",)


def test_batchnorm_running_statistics_are_checked():
    model = torch.nn.BatchNorm1d(3)
    ref = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.running_var.add_(0.25)
    out = attest(model, ref, required_keys=ref.keys())
    assert out.different == ("running_var",)
    assert "running_mean" in out.matched


def test_nonpersistent_buffer_cannot_be_attested_from_checkpoint():
    model = torch.nn.Module()
    model.register_buffer("runtime", torch.ones(3), persistent=False)
    out = attest(model, {"runtime": torch.ones(3)}, required_keys=("runtime",))
    assert not out.verified and out.missing_in_model == ("runtime",)


def test_tied_weights_verify_both_recorded_aliases():
    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 2, bias=False)
    model.b = torch.nn.Module()
    model.b.register_parameter("weight", model.a.weight)
    ref = copy.deepcopy(model.state_dict())
    out = attest(model, ref, required_keys=ref.keys())
    assert out.verified and len(out.matched) == 2


def test_tied_weight_missing_reference_alias_is_rejected():
    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 2, bias=False)
    model.b = torch.nn.Module()
    model.b.register_parameter("weight", model.a.weight)
    ref = copy.deepcopy(model.state_dict())
    del ref["b.weight"]
    out = attest(model, ref, required_keys=("a.weight", "b.weight"))
    assert out.missing_in_reference == ("b.weight",)


def test_signed_zero_bit_pattern_is_checked():
    model = torch.nn.Module()
    model.register_buffer("x", torch.tensor([0.0], dtype=torch.float32))
    ref = {"x": torch.tensor([-0.0], dtype=torch.float32)}
    out = attest(model, ref, required_keys=("x",))
    assert not out.verified and out.different == ("x",)


def test_equal_nan_bits_still_match():
    model = torch.nn.Module()
    model.register_buffer("x", torch.tensor([float("nan")], dtype=torch.float32))
    ref = copy.deepcopy(model.state_dict())
    assert attest(model, ref, required_keys=("x",)).verified


def test_bfloat16_storage_bytes_match():
    model = torch.nn.Module()
    model.register_buffer("x", torch.ones(4, dtype=torch.bfloat16))
    ref = copy.deepcopy(model.state_dict())
    assert attest(model, ref, required_keys=("x",)).verified


def test_lazy_unmaterialized_weight_is_unverifiable():
    model = torch.nn.LazyLinear(3)
    ref = {"weight": torch.zeros(3, 2)}
    out = attest(model, ref, required_keys=("weight",))
    assert not out.verified and out.unverifiable == ("weight",)


def test_meta_tensor_rejected():
    model = torch.nn.Module()
    model.register_buffer("x", torch.ones(2, device="meta"))
    ref = {"x": torch.ones(2, device="meta")}
    out = attest(model, ref, required_keys=("x",))
    assert not out.verified and out.unverifiable == ("x",)


def test_string_scope_rejected():
    model, ref = linear()
    with pytest.raises(TypeError, match="required_keys"):
        attest(model, ref, required_keys="weight")


def test_empty_scope_rejected():
    model, ref = linear()
    with pytest.raises(ValueError, match="required_keys"):
        attest(model, ref, required_keys=[])


def test_digest_is_deterministic_independent_of_input_key_order():
    model, ref = linear()
    a = attest(model, ref, required_keys=["weight", "bias"])
    b = attest(model, ref, required_keys=["bias", "weight"])
    assert a.verified and b.verified
    assert a.comparison_sha256 == b.comparison_sha256


def test_digest_changes_for_other_valid_bytes():
    model, ref = linear()
    a = attest(model, ref, required_keys=["weight", "bias"])
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        model2.weight.add_(1.0)
    ref2 = copy.deepcopy(model2.state_dict())
    b = attest(model2, ref2, required_keys=["weight", "bias"])
    assert a.comparison_sha256 != b.comparison_sha256


def test_live_state_dict_alias_cannot_certify_its_own_contents():
    model = torch.nn.Linear(2, 2)
    ref = model.state_dict()
    out = attest(model, ref, required_keys=ref.keys())
    assert not out.verified and sorted(out.unverifiable) == ["bias", "weight"]


def test_clone_reference_prevents_tautological_self_comparison():
    model = torch.nn.Linear(2, 2)
    ref = {name: value.clone() for name, value in model.state_dict().items()}
    assert attest(model, ref, required_keys=ref.keys()).verified
