"""Adversarial synthetic voltage-derivation tests; no participant data."""
from fractions import Fraction
import pytest
from trust_contract.signal_semantics import Channel, Manifest, classify, apply_transform


def ch(name, pos, neg):
    return Channel.difference(name, pos, neg)


def m(channels, **kw):
    fields = dict(acquisition_id="recording1", timebase_id="sample0",
                  sampling_hz="250", unit="uV")
    fields.update(kw)
    return Manifest(tuple(channels), **fields)


def test_exact_identity():
    x = m([ch("A-R", "A", "R")])
    assert classify(x, x)["status"] == "EXACT"




def test_exact_redundant_channel_basis_preserves_measured_identity():
    # A-B = (A-R) - (B-R) algebraically, but distinct measured traces may
    # carry their own sensor noise. EXACT must not substitute a reconstruction.
    source = m([
        ch("A-R", "A", "R"),
        ch("B-R", "B", "R"),
        ch("A-B", "A", "B"),
    ])
    report = classify(source, source)
    assert report["status"] == "EXACT"
    assert report["reason"] == "identity"
    assert report["matrix"] == [
        ["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"],
    ]
    # Deliberately independent recorded channel noise makes this regression
    # observable: arithmetic reconstruction of A-B would change the trace.
    signals = [[10.0, 20.0], [3.0, 4.0], [7.1, 16.2]]
    assert apply_transform(report["matrix"], signals) == signals


def test_channel_permutation():
    src = m([ch("A-R", "A", "R"), ch("B-R", "B", "R")])
    dst = m([ch("B-R", "B", "R"), ch("A-R", "A", "R")])
    assert classify(src, dst)["matrix"] == [["0", "1"], ["1", "0"]]


def test_redundant_channel_permutation_prefers_observed_lead():
    source = m([
        ch("A-R", "A", "R"),
        ch("B-R", "B", "R"),
        ch("A-B", "A", "B"),
    ])
    destination = m([
        ch("A-B", "A", "B"),
        ch("A-R", "A", "R"),
        ch("B-R", "B", "R"),
    ])
    report = classify(source, destination)
    assert report["status"] == "TRANSFORMABLE"
    assert report["matrix"] == [
        ["0", "0", "1"], ["1", "0", "0"], ["0", "1", "0"],
    ]
    measured = [[10.0, 20.0], [3.0, 4.0], [7.1, 16.2]]
    assert apply_transform(report["matrix"], measured) == [
        measured[2], measured[0], measured[1],
    ]


def test_common_reference_cancellation():
    src = m([ch("Fp1-Ref", "Fp1", "Ref"), ch("F7-Ref", "F7", "Ref")])
    dst = m([ch("Fp1-F7", "Fp1", "F7")])
    result = classify(src, dst)
    assert result["status"] == "TRANSFORMABLE"
    assert result["matrix"] == [["1", "-1"]]
    assert apply_transform(result["matrix"], [[10.0, 20.0], [3.0, 4.0]]) == [[7.0, 16.0]]


def test_polarity_inversion_explicit():
    assert classify(m([ch("A-B", "A", "B")]), m([ch("B-A", "B", "A")]))["matrix"] == [["-1"]]


def test_bipolar_cannot_be_inverted_into_monopolar():
    assert classify(m([ch("A-B", "A", "B")]), m([ch("A-R", "A", "R")]))["status"] == "INCOMPATIBLE"


def test_two_hop_bipolar_chain():
    src = m([ch("A-B", "A", "B"), ch("B-C", "B", "C")])
    assert classify(src, m([ch("A-C", "A", "C")]))["matrix"] == [["1", "1"]]


def test_different_reference_not_assumed_equal():
    assert classify(m([ch("A-R1", "A", "R1")]), m([ch("A-R2", "A", "R2")]))["status"] == "INCOMPATIBLE"


def test_midpoint_not_a_voltage_difference():
    assert classify(m([ch("A-B", "A", "B")]), m([ch("Mid-R", "Mid", "R")]))["status"] == "INCOMPATIBLE"


def test_unit_conversion():
    r = classify(m([ch("A-R", "A", "R")], unit="mV"),
                 m([ch("A-R", "A", "R")], unit="uV"))
    assert r["status"] == "TRANSFORMABLE" and r["matrix"] == [["1000"]]


def test_sampling_rate_change_rejected():
    assert classify(m([ch("A-R", "A", "R")], sampling_hz="250"),
                    m([ch("A-R", "A", "R")], sampling_hz="256"))["status"] == "INCOMPATIBLE"


def test_source_and_timebase_mismatch():
    x = m([ch("A-R", "A", "R")])
    assert classify(x, m([ch("A-R", "A", "R")], acquisition_id="other"))["status"] == "INCOMPATIBLE"
    assert classify(x, m([ch("A-R", "A", "R")], timebase_id="other"))["status"] == "INCOMPATIBLE"


@pytest.mark.parametrize("missing", ["acquisition_id", "timebase_id", "sampling_hz", "unit"])
def test_missing_provenance_is_unknown(missing):
    assert classify(m([ch("A-R", "A", "R")], **{missing: None}),
                    m([ch("A-R", "A", "R")]))["status"] == "UNKNOWN"


def test_channel_name_is_not_derivation_metadata():
    assert classify(m([Channel.unknown("A-B")]), m([ch("A-B", "A", "B")]))["status"] == "UNKNOWN"


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        m([ch("A-R", "A", "R"), ch("A-R", "A", "R")])


def test_nonzero_contrast_required():
    with pytest.raises(ValueError, match="reference-invariant"):
        m([Channel("A", (("A", Fraction(1)),))])


def test_ragged_sample_arrays_rejected():
    with pytest.raises(ValueError, match="ragged"):
        apply_transform([["1", "-1"]], [[1.0], [1.0, 2.0]])


def test_mne_bipolar_parity_synthetic():
    np = pytest.importorskip("numpy")
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(42)
    src = rng.normal(0, 1e-5, (2, 128))
    raw = mne.io.RawArray(src, mne.create_info(["A-R", "B-R"], 250, "eeg"), verbose=False)
    want = mne.set_bipolar_reference(raw, "A-R", "B-R", copy=True, verbose=False).get_data()[0]
    matrix = classify(m([ch("A-R", "A", "R"), ch("B-R", "B", "R")], unit="V"),
                      m([ch("A-B", "A", "B")], unit="V"))["matrix"]
    np.testing.assert_allclose(apply_transform(matrix, src.tolist())[0], want, rtol=1e-12, atol=1e-18)


def test_basis_random_linear_combinations_exact():
    import random
    rng = random.Random(20261117)
    for trial in range(45):
        rows = []
        for i in range(6):
            weights = [rng.randint(-2, 2) for _ in range(5)]
            if not any(weights):
                weights[0] = 1
            rows.append(weights)
        source = [Channel(f"src_{i}", tuple((f"E{j}", Fraction(v)) for j,v in enumerate(row) if v)
                  + (("REF", Fraction(-sum(row))),)) for i, row in enumerate(rows)]
        combo = [rng.randint(-2, 2) for _ in rows]
        target_coeffs = [sum(c*row[j] for c,row in zip(combo, rows)) for j in range(5)]
        if not any(target_coeffs):
            continue
        target = Channel("target", tuple((f"E{j}", Fraction(v)) for j,v in enumerate(target_coeffs) if v)
                         + (("REF", Fraction(-sum(target_coeffs))),))
        report = classify(m(source), m([target]))
        assert report["status"] == "TRANSFORMABLE", trial
        reconstructed = {}
        for row, coefficient in zip(source, report["matrix"][0]):
            for node, value in row.coefficients().items():
                reconstructed[node] = reconstructed.get(node, Fraction()) + Fraction(coefficient)*value
        assert {k:v for k,v in reconstructed.items() if v} == target.coefficients(), trial


def test_basis_refuses_unobserved_independent_electrode():
    source = m([ch("A-R", "A", "R"), ch("B-R", "B", "R")])
    target = m([ch("C-R", "C", "R")])
    assert classify(source, target)["status"] == "INCOMPATIBLE"
