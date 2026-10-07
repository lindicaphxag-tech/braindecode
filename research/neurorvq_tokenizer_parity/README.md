# NeuroRVQ tokenizer reference-parity evidence

This package independently checks the Braindecode `NeuroRVQTokenizer` against the released NeuroRVQ implementation and checkpoint. It was rerun after the Braindecode maintainers refactored the tokenizer to reuse shared layers and removed unused quantizer losses.

## Pinned inputs

- Braindecode PR #1223 source: `61194d513e0c5234a2f55923bf157643354e4b42`.
- Official NeuroRVQ source: `926e770d9d16b6aa308404280fa0cc0211a6f9fb`.
- Hugging Face checkpoint revision: `d944b87f44ae0ba2923b2f10d0518f23f6803b76`.
- Checkpoint: `pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt`, 304,130,106 bytes, SHA-256 `d3e850e160b1a558c529466b1c0b8cebe8888a5e02824979886dd226187b2725`.

The checkpoint is downloaded from Hugging Face by the runner unless a local `--checkpoint` path is supplied. It is not redistributed here.

## Method

The script builds both implementations from the same pinned checkpoint and feeds an identical seeded synthetic tensor with shape `[2, 3, 400]` (channels F3/F4/Cz, 200 Hz, two 200-sample patches). It checks evaluation targets and reconstructions, discrete token IDs, training-mode targets and reconstructions, all trainable parameter gradients, input gradients, and EMA codebook state after one training-mode forward update. MLP parameter names are normalized only for the reference-versus-Braindecode name comparison (`mlp.fc1/fc2` versus sequential `mlp.0/2`); tensor values and gradients are compared directly.

## Reproduce

Use a checkout of this evidence branch, install Braindecode's runtime dependencies and `huggingface_hub`, and clone the official source at its pinned commit:

```bash
git clone https://github.com/lindicaphxag-tech/braindecode.git braindecode
cd braindecode
git checkout research/neurorvq-tokenizer-parity
python -m pip install -e . huggingface_hub
git clone https://github.com/KonstantinosBarmpas/NeuroRVQ.git ../NeuroRVQ
git -C ../NeuroRVQ checkout 926e770d9d16b6aa308404280fa0cc0211a6f9fb
python research/neurorvq_tokenizer_parity/validate_neurorvq_tokenizer_parity.py \
  --neurorvq-source ../NeuroRVQ \
  --output-json research/neurorvq_tokenizer_parity/results/rerun.json
```

The checkpoint repository revision and source revision are pinned in the script. The run record and hashes for the recorded result are in `results/reference_parity_61194d5.json` and `manifest.json`.

## Recorded result

Run environment: Python 3.12.4, PyTorch 2.8.0+cpu, NumPy 2.2.6, Windows 11, CPU execution. Seed: 43. CUDA was unavailable.

| Check | Result |
|---|---:|
| Evaluation target max absolute error | 0 |
| Evaluation reconstruction max absolute error | 0 |
| Discrete token IDs | exactly equal |
| Training target max absolute error | 0 |
| Training reconstruction max absolute error | 0 |
| Max parameter-gradient error across 345 parameters | `7.11e-15` |
| Input-gradient max absolute error | 0 |
| EMA state after one training-mode update | exactly equal |

## Limits

This is a deterministic implementation-parity check for one synthetic input shape, one checkpoint, and one software environment. It does not reproduce the paper's dataset benchmarks, estimate task accuracy, test other channel layouts or random seeds, establish robustness, or demonstrate clinical utility. The upstream PR and this evidence branch do not claim those results or SCI-Q1 status.

## High Gamma Table 10 replication

`reproduce_table10_highgamma.py` adds a separate dataset-level check for the raw-signal reconstruction MSE reported for High Gamma in Table 10 of the NeuroRVQ paper. It evaluates all 14 Schirrmeister 2017 subjects by default with the pinned checkpoint and the Braindecode implementation at commit `61194d513e0c5234a2f55923bf157643354e4b42`.

The script follows the public [EEG-Benchmarking preprocessing protocol](https://github.com/dykestra/EEG-Benchmarking): 0.5–45 Hz filtering, resampling to 200 Hz, four-second cue-locked trials, the documented non-standard channel exclusions, and common-average reference. It reports both trial-pooled and subject-macro MSE, per-subject results, and filtered-band diagnostics. The paper's High Gamma raw-signal target is 0.084; the script does not treat a partial-subject smoke run as a reproduction.

The Kaggle kernel is configured as private and requests a T4 GPU. Run it from this directory:

```bash
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status oblivicore/neurorvq-table-10-high-gamma-replication
kaggle kernels output oblivicore/neurorvq-table-10-high-gamma-replication \
  --file-pattern '^(experiment_log|artifacts_manifest|metrics)\\.jsonl?$|^per_subject\\.csv$' \
  -p remote_outputs/neurorvq-table10
```

The raw EDF cache and downloaded checkpoint are stored under `/kaggle/temp`; only aggregate JSON/CSV evidence is written under `/kaggle/working`. The HGD dataset is CC BY 4.0 and the upstream checkpoint/source has CC BY-NC 4.0 terms. Neither raw EEG nor checkpoint files belong in the public evidence branch. A successful full 14-subject run, with its retrieved metrics and exact run manifest, is still required before calling the Table 10 number reproduced.
