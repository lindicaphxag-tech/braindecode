# RFC: semantic channel compatibility (external proposal)

## Technical question

Braindecode's merged [#1241](https://github.com/braindecode/braindecode/pull/1241) standardizes geometric channel strategies. It does not by itself establish whether an EEG channel measures the *same voltage derivation* as a checkpoint expects. This extension is motivated by [#995](https://github.com/braindecode/braindecode/issues/995).

## Formal contract

For physically aligned electrode potentials v and channel contrast matrices C_src and C_target (zero-sum rows), find a rational matrix A satisfying A C_src = C_target. Adjust numerical units by u_src/u_target, where each u is volts per unit. Report an exact mapping only for declared same acquisition/timebase and sampling frequency.

Status taxonomy: EXACT (identity), TRANSFORMABLE (explicit signed/rational mapping), INCOMPATIBLE (no exact mapping), UNKNOWN (insufficient metadata). All transformations are returned explicitly; the library never silently swaps polarity.

| Case | Outcome |
|---|---|
| [Fp1-Ref, F7-Ref] -> [Fp1-F7] | TRANSFORMABLE [1,-1] |
| Fp1-F7 -> F7-Fp1 | TRANSFORMABLE [-1] |
| Fp1-F7 -> Fp1-Ref | INCOMPATIBLE |
| 250 Hz -> 256 Hz | INCOMPATIBLE without explicit resampling |
| Channel name only, missing actual derivation | UNKNOWN |

## Existing software boundary

This is a conservative *preflight* and not another channel interpolation layer, not a model API replacement, and not a guarantee of pretrained feature accuracy. Real data may differ in electrode placement, transfer functions, acquisition filters or physical referencing even when metadata names match.

## Proposed upstream review questions

1. Prefer a test-only helper, a serialized model-card acquisition manifest, or an opt-in public API?
2. How to encode electrode identity, reference/derivation, unit and sample timebase with MNE/BIDS metadata?
3. Should frozen feature extraction require stricter policy than fine-tuning?
4. Which actual released checkpoint should serve as the first pinned reproduction?

The current code is staged in the user's private research repo. Do not merge into the Braindecode project or change model defaults before maintainers accept the scope.
