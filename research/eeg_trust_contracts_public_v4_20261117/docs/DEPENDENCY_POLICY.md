# Dependency and reproducibility policy

The EEG signal-derivation and evaluation-metadata preflight code is pure Python and must be importable without PyTorch. The pretrained-parameter audit requires PyTorch and is **loaded lazily** via `from trust_contract import audit_model`.

## Installation modes

- Semantics/metadata only: `pip install -e .`
- Full CPU test suite: `pip install -e '.[tests]'`
- Real pretrained-origin audit (if user authorizes downloading checkpoints): `pip install -e '.[pretrained]'`

For this research package, run commands from `research/eeg_trust_contracts_20261117`. The optional tests extra explicitly includes PyTorch; without the extra, the lightweight signal/evaluation API does not depend on it.

## Regression evidence

- The package previously imported PyTorch in `trust_contract.__init__` without declaring it under `project.dependencies`.
- A subprocess test inserts a meta-path finder that raises `ModuleNotFoundError` for all `torch` imports. Under that simulation, `trust_contract`, `trust_contract.signal_semantics`, `trust_contract.evaluation_audit` and the offline CLI import successfully.
- A second test ensures the optional `audit_model` symbol resolves when PyTorch is installed.
- The corresponding locally executed suite: **68 passed** (2026-11-17), including eight physical-site lineage cases. Official private GitHub Actions are separately affected by runner allocation failures; no cloud green result is claimed.

All research prototypes remain independent of Braindecode official releases.
