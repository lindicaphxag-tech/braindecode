"""Conservative channel-derivation compatibility checker.

Signals are specified by symbolic electrode-potential differences. An exact
mapping is certified ONLY for the declared same acquisition and sample timebase,
known electrode derivations, identical sampling rates, and compatible units.

This is a mathematical preflight check, NOT empirical montage equivalence or
an automatic replacement for MNE's signal conversion operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

Unit = Literal["V", "mV", "uV"]
_UNIT_V = {"V": Fraction(1), "mV": Fraction(1, 1000), "uV": Fraction(1, 1_000_000)}


def _fraction(value: int | str | Fraction) -> Fraction:
    if isinstance(value, float):
        raise TypeError("use integer, Fraction or decimal string for exact rational metadata")
    return Fraction(value)


@dataclass(frozen=True)
class Channel:
    """One voltage difference expressed as a linear combination of potentials.

    terms=None means the physical derivation is unknown. An empty term list is
    rejected, because it is not a measurable potential difference.
    """
    name: str
    terms: tuple[tuple[str, Fraction], ...] | None

    @classmethod
    def difference(cls, name: str, positive: str, negative: str) -> "Channel":
        if not positive or not negative or positive == negative:
            raise ValueError("a difference needs distinct nonempty electrode labels")
        return cls(name, ((positive, Fraction(1)), (negative, Fraction(-1))))

    @classmethod
    def unknown(cls, name: str) -> "Channel":
        return cls(name, None)

    def coefficients(self) -> dict[str, Fraction] | None:
        if self.terms is None:
            return None
        if not self.name or not self.terms:
            raise ValueError("a known channel needs nonempty name and terms")
        result: dict[str, Fraction] = {}
        for electrode, coeff in self.terms:
            if not electrode:
                raise ValueError("electrode identifiers must be explicit")
            result[electrode] = result.get(electrode, Fraction()) + _fraction(coeff)
        result = {key: value for key, value in result.items() if value}
        if not result or sum(result.values()) != 0:
            raise ValueError("voltage derivations must be nonzero, reference-invariant contrasts")
        return result


@dataclass(frozen=True)
class Manifest:
    channels: tuple[Channel, ...]
    acquisition_id: str | None
    timebase_id: str | None
    sampling_hz: str | None
    unit: Unit | None

    def __post_init__(self):
        if not self.channels or len({c.name for c in self.channels}) != len(self.channels):
            raise ValueError("channels must be nonempty and have unique names")
        if any(not c.name for c in self.channels):
            raise ValueError("channel names cannot be empty")
        for c in self.channels:
            c.coefficients()
        if self.sampling_hz is not None and _fraction(self.sampling_hz) <= 0:
            raise ValueError("sampling rate must be positive")
        if self.unit is not None and self.unit not in _UNIT_V:
            raise ValueError("unknown physical unit")


def _rational_solve(source: list[list[Fraction]], target: list[Fraction]) -> list[Fraction] | None:
    """Find c with sum(c_i * source_i) = target; exact Gauss-Jordan."""
    n_src = len(source)
    n_features = len(target)
    if n_src == 0:
        return None
    augmented = [[source[col][row] for col in range(n_src)] + [target[row]] for row in range(n_features)]
    pivots: list[tuple[int, int]] = []
    row_idx = 0
    for col_idx in range(n_src):
        pivot = next((r for r in range(row_idx, n_features) if augmented[r][col_idx]), None)
        if pivot is None:
            continue
        augmented[row_idx], augmented[pivot] = augmented[pivot], augmented[row_idx]
        div = augmented[row_idx][col_idx]
        augmented[row_idx] = [v / div for v in augmented[row_idx]]
        for r in range(n_features):
            if r != row_idx:
                factor = augmented[r][col_idx]
                if factor:
                    augmented[r] = [v - factor * p for v, p in zip(augmented[r], augmented[row_idx])]
        pivots.append((row_idx, col_idx))
        row_idx += 1
        if row_idx == n_features:
            break
    for row in augmented:
        if all(v == 0 for v in row[:-1]) and row[-1] != 0:
            return None
    coefficients = [Fraction() for _ in range(n_src)]
    for r, c in pivots:
        coefficients[c] = augmented[r][-1]
    return coefficients


def classify(source: Manifest, target: Manifest) -> dict:
    """Return UNKNOWN, INCOMPATIBLE, EXACT, or TRANSFORMABLE.

    TRANSFORMABLE returns an explicit channel transformation. No implicit
    polarity inversion, resampling, interpolation, or reference assumptions.
    """
    if not all((source.acquisition_id, target.acquisition_id, source.timebase_id, target.timebase_id,
                source.sampling_hz, target.sampling_hz, source.unit, target.unit)):
        return {"status": "UNKNOWN", "reason": "missing acquisition/timebase/sampling/unit provenance", "matrix": None}
    if source.acquisition_id != target.acquisition_id or source.timebase_id != target.timebase_id:
        return {"status": "INCOMPATIBLE", "reason": "different acquisition or sample timebase", "matrix": None}
    if _fraction(source.sampling_hz) != _fraction(target.sampling_hz):
        return {"status": "INCOMPATIBLE", "reason": "sample rates differ; no implicit resampling", "matrix": None}
    all_coeffs = [c.coefficients() for c in (*source.channels, *target.channels)]
    if any(x is None for x in all_coeffs):
        return {"status": "UNKNOWN", "reason": "at least one channel derivation is undocumented", "matrix": None}
    nodes = sorted({node for formula in all_coeffs for node in formula})
    src = [[formula.get(n, Fraction()) for n in nodes] for formula in all_coeffs[:len(source.channels)]]
    tgt = [[formula.get(n, Fraction()) for n in nodes] for formula in all_coeffs[len(source.channels):]]
    unit_ratio = _UNIT_V[source.unit] / _UNIT_V[target.unit]
    matrix: list[list[Fraction]] = []
    for desired in tgt:
        coefficients = _rational_solve(src, desired)
        if coefficients is None:
            return {"status": "INCOMPATIBLE", "reason": "target derivation is outside source signal span", "matrix": None}
        matrix.append([coefficient * unit_ratio for coefficient in coefficients])
    direct = (len(source.channels) == len(target.channels)
              and all(s.name == t.name and s.coefficients() == t.coefficients()
                      for s, t in zip(source.channels, target.channels))
              and source.unit == target.unit)
    return {"status": "EXACT" if direct else "TRANSFORMABLE",
            "reason": "identity" if direct else "explicit exact rational linear derivation",
            "matrix": [[str(x) for x in row] for row in matrix]}


def apply_transform(matrix: list[list[str]], samples: list[list[float]]) -> list[list[float]]:
    """Transform source-channel-major samples; explicit signed coefficients."""
    if not matrix:
        raise ValueError("empty transform matrix")
    count = len(samples)
    if not count or not samples[0] or any(len(s) != len(samples[0]) for s in samples):
        raise ValueError("ragged or empty source samples")
    if any(len(row) != count for row in matrix):
        raise ValueError("matrix input channels and samples disagree")
    return [[float(sum(Fraction(c) * Fraction(str(samples[j][t])) for j, c in enumerate(row)))
             for t in range(len(samples[0]))] for row in matrix]
