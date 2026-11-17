# EEG Foundation Model Trust Contracts — Public Research Prototype

**By Zhibo Zhang (@lindicaphxag-tech). Not part of the official Braindecode package or release.**

This independent research prototype implements three distinct checkers:

1. **EEG channel-derivation semantics:** exact rational signed-voltage contrasts for electrode polarity, reference identity, units, recording timebase, and transformations. Includes MNE synthetic-reference parity tests. No unknown-reference inference.
2. **Evaluation integrity:** train/validation/test subject, recording, optional physical-site and dataset identity checks. Explicit global IDs only; missing cross-dataset physical-site mappings block strict site-held-out certification.
3. **Pretrained tensor-origin audit:** read-only loaded/new/adapted/lazy tensor checks, plus optional version-pinned Hub checkpoint probe. No real released-checkpoint numerical fidelity or EEG benchmark claim is made.

## CPU reproduction

From this directory:
```bash
python -m pip install -e '.[tests]'
python -m pytest -q tests
python -m trust_contract.cli signal --input examples/semantic_case.json
python -m trust_contract.cli evaluation --input examples/site_holdout_unknown.json
```

The channel/evaluation utilities import **without PyTorch**, which is a declared optional dependency for pretrained auditing. For Hub downloads, explicit opt-in and exact commit revisions are required.

**Evidence scope:** The corresponding locally executed source package passed 68 synthetic CPU tests on 2026-11-17 (includes MNE synthetic bipolar comparisons and subprocess test with PyTorch blocked). This public GitHub source mirror has not separately received a verified GitHub Actions pass. Source was developed in private [PR #86](https://github.com/lindicaphxag-tech/cs/pull/86) and [PR #87](https://github.com/lindicaphxag-tech/cs/pull/87).

Related Braindecode discussions: [#995](https://github.com/braindecode/braindecode/issues/995), [#1227](https://github.com/braindecode/braindecode/issues/1227), and separate upstream Draft [#1574](https://github.com/braindecode/braindecode/pull/1574).

**Limitations:** This is not proof of clinical safety, EEG generalization, checkpoint reference parity, upstream adoption or maintainer status. Do not put real participant or hospital identifiers in public manifests.
