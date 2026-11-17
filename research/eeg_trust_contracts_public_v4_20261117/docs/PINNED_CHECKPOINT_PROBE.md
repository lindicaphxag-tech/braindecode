# Pinned Braindecode checkpoint-origin probe (opt-in)

This is an **unmerged research runner**, not a claim of real pretrained model fidelity.

It bridges the metadata-only v2 `pretrained_audit` helper with one actual Braindecode `from_pretrained` invocation. By default there is no network. Loading from a Hub repository requires both `--allow-download` and an exact 40-hex commit revision. All local or downloaded `.safetensors` files are hashed with SHA-256. JSON output records loaded/adapted/new/uninitialized parameters and explicitly declared feature-scope risks.

## Reproduction

Pin Braindecode, `huggingface_hub`, PyTorch and safetensors versions in an isolated environment. For a checkpoint already on disk:

```bash
PYTHONPATH=research/eeg_trust_contracts_20261117/src python -m trust_contract.checkpoint_probe \
  --snapshot /path/to/pinned/snapshot \
  --model-class BIOT \
  --expected-new final_layer. \
  --feature-scope encoder. \
  --freeze --output origin-report.json
```

The example class and feature-scope must be adapted to the actual model tree and verified forward path. No validity of `encoder.` as a universal BIOT feature scope is claimed. For a Hub download use `--repo-id` instead of `--snapshot`, and pass both `--revision <40-character SHA>` and `--allow-download`. Review model license/access restrictions first.

## Limitations

- A matching weight hash does not establish correspondence with the original author's checkpoint or forward implementation.
- Loaded tensors do not prove correct graph usage or reference fidelity.
- Correct channel shapes do not prove matching physical electrode derivation.
- The probe does not measure accuracy, clinical validity or subject-held-out generalization.

## Verification

```bash
PYTHONPATH=research/eeg_trust_contracts_20261117/src python -m pytest -q research/eeg_trust_contracts_20261117/tests/test_checkpoint_probe.py
```

A corresponding local test implementation passed four no-network CPU tests with a stub model and synthetic file bytes. **No real Hub checkpoint was downloaded or verified.** The GitHub CI state should be checked separately before claiming official CI acceptance.

## Next gate

Select a released licensed model, pin a real Hub commit and original source reference, verify the complete parameter origin report on a real snapshot, then compare predeclared intermediate activations on a fixed EEG recording. Report mismatches and negative results, not just positive parity.
