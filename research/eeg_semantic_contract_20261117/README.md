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

## Solver update

The public research branch now factors the source derivation basis once and reuses it for all target channels; the synthetic 128-channel CPU preflight microbenchmark had a three-run median of about 0.134 seconds on the tested machine. This is an offline metadata-validation timing, **not** a Braindecode model speedup or a general hardware performance guarantee. Independent real-model checks remain pending.

## Observed-channel fidelity

The rational solver now prioritizes channels that are **already measured with the same declared name and voltage contrast**. In a redundant montage such as `[A-R, B-R, A-B]`, keeping the exact input order returns the identity matrix; reordering those same inputs returns a direct permutation, not an unnecessary reconstruction of `A-B` from `(A-R)-(B-R)`. The ideal contrasts are equivalent, but independently recorded traces can differ through channel-specific noise. Two synthetic regressions cover identity and permutation in this redundant setting.

This behavior is confined to the external research branch; no Braindecode upstream API, checkpoint or #1577 code was changed. GitHub Actions for this research-branch commit did not report a workflow run at the time of this update; the main Braindecode PR's successful CI is separate.

## Cross-evidence validation gate (research only)

`src/trust_contract/validation_policy.py` provides a **read-only, opt-in evidence decision** layer that consumes, rather than reimplements, the upstream capabilities already merged in Braindecode:
- #1241 for existing channel strategies;
- #1493 for model-card pretraining corpus overlap information;
- #1558 for pretrained parameter-origin records; and
- #1208 for the registry-wide runtime model contract checks.

The policy has four outcomes: `ACCEPT` (the supplied evidence meets the narrowly declared checks), `DISCLOSE` (an in-corpus or incompletely traced claim requires a label), `REVIEW` (metadata, feature path, or an applied channel transform is not verified), `REFUSE` (known incompatibility, a claimed held-out overlap, or unloaded parameters in a declared frozen feature scope).

**No inference of model control flow is attempted.** Callers must supply the concrete feature path and actual loaded-parameter names; this is not a second `_pretrained_provenance` implementation and not a general proof that real EEG recordings match checkpoint pretraining. The explicit `channel_transform_applied` flag requires the mapping to have been executed and checked, not just mathematically available.

Run locally without requiring Braindecode or downloading weights:

```bash
cd research/eeg_semantic_contract_20261117
python -m pip install -e '.[tests]'
python -m pytest -q tests/test_validation_policy.py
```

At development time, 30 local synthetic cases passed. This is a **research-branch test result, not official Braindecode CI**. The package has not loaded any released checkpoint and no upstream API/behavior was changed. A future upstream PR would require a maintainer-agreed scope and explicit native checkpoint evaluation first.

## Read-only bridge to existing checkpoint provenance

`src/trust_contract/model_bridge.py` reads the **already available**
Braindecode `_pretrained_provenance` dictionary and current
`named_parameters(remove_duplicate=False)` without installing hooks,
downloading weights or changing inference. It recognizes a pinned 40-hex
checkpoint commit; reconciles tied aliases by parameter object identity;
and excludes all aliases of mixed/adapted parameters from full-load coverage.

The caller must provide the audited feature-path parameter names and an
independently justified corpus overlap decision. Neither `model.card_data.datasets`
nor a successful state-dict load is treated as proof that a dataset was
held out or that all feature-path parameters came from the checkpoint.
An absent/unpinned provenance record forces `REVIEW`.

```python
from trust_contract.model_bridge import inspect_model

# model: already loaded by the ordinary Braindecode API
report = inspect_model(
    model,
    workload="frozen_features",
    signal_status="EXACT",  # independently certified manifest
    feature_scope={"encoder.weight"},  # independently audited actual graph
    overlap="UNKNOWN",  # do not infer disjointness from missing card metadata
    claim_held_out=True,
)
print(report.report.as_json())
```

Local checks after this addition: 44/44 validation and bridge cases passed,
including an actual PyTorch parameter-alias test when PyTorch was installed.
The GitHub research workflow also runs the pure-Python bridge tests, with
the optional torch check skipped when torch is absent. These are **not**
checkpoint inference tests and are **not** an upstream Braindecode CI result.
