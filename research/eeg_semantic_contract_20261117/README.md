# EEG signal semantic contract — external, opt-in research RFC

This public fork branch contains a small, isolated proof of a missing **physical channel derivation** guarantee discussed in [Braindecode #995](https://github.com/braindecode/braindecode/issues/995). It is **not** a Braindecode API, does not replace [#1241](https://github.com/braindecode/braindecode/pull/1241), and is not a claim about model performance.

Run without downloads other than test dependencies:

```bash
cd research/eeg_semantic_contract_20261117
python -m pip install -e '.[tests]'
python -m pytest -q tests/test_signal_semantics.py
```

The synthetic tests cover explicit bipolar-reference cancellation, polarity inversion, channel permutation, unit conversion, unknown reference, sample-rate/timebase mismatch, non-identifiable inversion, and a synthetic-waveform parity check against MNE's `set_bipolar_reference`.

**Key restriction:** only declared, physically aligned, same-acquisition/timebase voltage contrasts are eligible for an exact rational transform. Unknown derivations cannot be inferred from names; interpolation or resampling is not silently treated as exact. The acquisition metadata must be independently reliable.

A corresponding local source-package archive additionally contains pretrained parameter-origin auditing and evaluation split/corpus integrity tests. Those other modules and unsubmitted research materials remain separate, and are not prerequisites for reviewing this narrowly scoped semantic RFC.

Upstream PRs should be opened only once maintainers agree on interface placement and scope. Contact: Zhibo Zhang (@lindicaphxag-tech).
