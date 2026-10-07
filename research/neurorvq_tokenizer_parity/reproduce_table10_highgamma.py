"""Reproduce the NeuroRVQ High Gamma reconstruction number from Table 10.

The script downloads the public Schirrmeister 2017 High Gamma EDFs directly
from the dataset's GIN mirror, applies the preprocessing specified by the
EEG-Benchmarking repository, and evaluates the pinned released tokenizer
checkpoint through the Braindecode PR implementation. It writes aggregate and
per-subject results but never exports raw EEG or checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import scipy
import torch
from scipy.signal import butter, sosfiltfilt

_DATA_URL = "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
_DATA_LICENSE = "CC BY 4.0 (Schirrmeister 2017 High Gamma dataset)"
_BENCHMARK_URL = "https://github.com/dykestra/EEG-Benchmarking"
_BRAiNDECODE_COMMIT = "61194d513e0c5234a2f55923bf157643354e4b42"
_HF_REPO = "ntinosbarmpas/NeuroRVQ"
_HF_REVISION = "d944b87f44ae0ba2923b2f10d0518f23f6803b76"
_TARGET_MSE = 0.084
_SAMPLE_RATE = 200
_N_SAMPLES = 800
_BATCH_SIZE = int(os.environ.get("NEURORVQ_BATCH_SIZE", "16"))
_SUBJECTS: list[int] = []

_NON_EEG = {"eogh", "eogv", "emg_rh", "emg_lh", "emg_rf"}
_NON_STANDARD = {
    "aff1", "aff2", "ffc5h", "ffc3h", "ffc4h", "ffc6h", "fcc5h", "fcc3h",
    "fcc4h", "fcc6h", "ccp5h", "ccp3h", "ccp4h", "ccp6h", "cpp5h",
    "cpp3h", "cpp4h", "cpp6h", "ppo1", "ppo2", "i1", "i2", "afp3h",
    "afp4h", "aff5h", "aff6h", "fft7h", "ffc1h", "ffc2h", "fft8h",
    "ftt7h", "fcc1h", "fcc2h", "ftt8h", "ccp1h", "ccp2h", "ttp8h",
    "tpp7h", "cpp1h", "cpp2h", "ppo9h", "ppo5h", "ppo6h", "ppo10h",
    "poo9h", "poo3h", "poo4h", "poo10h", "oi1h", "oi2h",
}
_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _parse_subjects(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(bound) for bound in part.split("-", maxsplit=1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if not result or any(subject < 1 or subject > 14 for subject in result):
        raise ValueError("HGD_SUBJECTS must select subject IDs from 1 through 14.")
    return sorted(set(result))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1024:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroRVQ-replication/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if temporary.stat().st_size <= 1024:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")
    temporary.replace(path)


def _event_class(description: str) -> str | None:
    key = description.strip().lower().replace(" ", "_")
    if key.endswith(".0"):
        key = key[:-2]
    aliases = {
        "1": "right_hand", "t1": "right_hand", "right_hand": "right_hand",
        "right_fist": "right_hand", "2": "left_hand", "t2": "left_hand",
        "left_hand": "left_hand", "left_fist": "left_hand", "3": "rest",
        "t0": "rest", "rest": "rest", "no_action": "rest", "4": "feet",
        "t3": "feet", "feet": "feet", "both_feet": "feet",
    }
    return aliases.get(key)


def _read_recording(edf_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, infer_types=True, verbose="ERROR")
    raw_sampling_rate = float(raw.info["sfreq"])
    # Follow EEG-Benchmarking's documented 0.5-45 Hz bandpass and 200 Hz grid.
    raw.filter(l_freq=0.5, h_freq=45.0, method="fir", phase="zero", verbose="ERROR")
    raw.resample(_SAMPLE_RATE, npad="auto", verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, regexp=None, verbose="ERROR")

    selected = [
        index for index, name in enumerate(raw.ch_names)
        if name.strip().lower() not in _NON_EEG | _NON_STANDARD
    ]
    channel_names = [raw.ch_names[index].strip().lower() for index in selected]
    if len(channel_names) != len(set(channel_names)):
        raise ValueError(f"Duplicate normalized channel names in {edf_path}: {channel_names}")
    signal = raw.get_data(picks=selected).astype(np.float32, copy=False)
    signal -= signal.mean(axis=0, keepdims=True)  # match benchmark common-average reference

    annotation_by_code = {code: description for description, code in event_id.items()}
    windows: list[np.ndarray] = []
    labels: list[int] = []
    label_index = {"rest": 0, "left_hand": 1, "right_hand": 2, "feet": 3}
    skipped_boundary = 0
    for sample, _, code in events:
        cls = _event_class(annotation_by_code[int(code)])
        if cls is None:
            continue
        start = int(sample - raw.first_samp)
        stop = start + _N_SAMPLES
        if start < 0 or stop > signal.shape[1]:
            skipped_boundary += 1
            continue
        windows.append(signal[:, start:stop])
        labels.append(label_index[cls])
    if not windows:
        raise RuntimeError(
            f"No HGD class events found in {edf_path}. Annotation descriptions: "
            f"{sorted(event_id)}"
        )
    metadata = {
        "recording": edf_path.name,
        "n_events": len(events),
        "n_trials": len(windows),
        "skipped_boundary_events": skipped_boundary,
        "raw_sampling_rate_hz": raw_sampling_rate,
        "processed_sampling_rate_hz": float(raw.info["sfreq"]),
        "channels": channel_names,
        "mapped_classes": {name: sum(label == index for label in labels)
                           for index, name in enumerate(("rest", "left_hand", "right_hand", "feet"))},
        "event_descriptions": sorted(event_id),
    }
    return np.stack(windows), np.asarray(labels, dtype=np.int64), channel_names, metadata


def _filter_band(batch: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    sos = butter(4, band, btype="bandpass", fs=_SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, batch, axis=-1).astype(np.float32, copy=False)


def _evaluate(
    model, trials: np.ndarray, device: torch.device, n_channels: int
) -> dict[str, object]:
    raw_errors: list[np.ndarray] = []
    band_errors: dict[str, list[np.ndarray]] = {name: [] for name in _BANDS}
    for start in range(0, len(trials), _BATCH_SIZE):
        batch_np = trials[start : start + _BATCH_SIZE]
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            target, reconstruction = model(batch)
        target_np = target.detach().cpu().numpy().reshape(len(batch_np), n_channels, _N_SAMPLES)
        reconstruction_np = reconstruction.detach().cpu().numpy().reshape(target_np.shape)
        raw_errors.append(((target_np - reconstruction_np) ** 2).mean(axis=(1, 2)))
        for name, band in _BANDS.items():
            target_band = _filter_band(target_np, band)
            reconstruction_band = _filter_band(reconstruction_np, band)
            band_errors[name].append(((target_band - reconstruction_band) ** 2).mean(axis=(1, 2)))
    return {
        "trial_count": len(trials),
        "raw_mse_by_trial": np.concatenate(raw_errors),
        "band_mse_by_trial": {name: np.concatenate(values) for name, values in band_errors.items()},
    }


def main() -> None:
    global _SUBJECTS
    _SUBJECTS = _parse_subjects(os.environ.get("HGD_SUBJECTS", "1-14"))
    output = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/neurorvq_table10"))
    data_root = Path(os.environ.get("HGD_DATA_DIR", "/kaggle/temp/high_gamma_edf"))
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.time()
    log: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "braindecode_commit": _BRAiNDECODE_COMMIT,
        "checkpoint": {"repo_id": _HF_REPO, "revision": _HF_REVISION},
        "dataset": {"name": "Schirrmeister 2017 High Gamma", "license": _DATA_LICENSE,
                    "source": _DATA_URL, "benchmark_protocol": _BENCHMARK_URL,
                    "subjects": _SUBJECTS},
        "protocol": {"filter_hz": [0.5, 45.0], "resample_hz": _SAMPLE_RATE,
                     "epoch_seconds": 4, "common_average_reference": True,
                     "input_normalization": "Braindecode model per-trial/channel/patch z-score",
                     "band_mse_filter": "4th-order Butterworth SOS, zero-phase; 1-4, 4-8, 8-13, 13-30, 30-45 Hz",
                     "batch_size": _BATCH_SIZE, "seed": 43},
        "published_table10_hgd_raw_mse": _TARGET_MSE,
        "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                      "torch": torch.__version__, "mne": mne.__version__, "numpy": np.__version__,
                      "scipy": scipy.__version__, "cuda_available": torch.cuda.is_available()},
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This replication kernel requires the requested Kaggle GPU runtime.")
        from braindecode.models import NeuroRVQTokenizer
        from huggingface_hub import hf_hub_download

        checkpoint_root = Path(os.environ.get("HF_HOME", "/kaggle/temp/hf_cache"))
        os.environ["HF_HOME"] = str(checkpoint_root)
        checkpoint_path = Path(hf_hub_download(
            repo_id=_HF_REPO,
            filename="pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt",
            revision=_HF_REVISION,
        ))
        channel_names: list[str] | None = None
        all_trials: list[np.ndarray] = []
        all_subjects: list[int] = []
        all_trial_labels: list[int] = []
        recording_metadata: list[dict[str, object]] = []
        for subject in _SUBJECTS:
            for split in ("train", "test"):
                path = data_root / split / f"{subject}.edf"
                _download(f"{_DATA_URL}/{split}/{subject}.edf", path)
                trials, labels, names, meta = _read_recording(path)
                if channel_names is None:
                    channel_names = names
                elif channel_names != names:
                    raise ValueError(f"Channel order differs for subject {subject} {split}.")
                all_trials.append(trials)
                all_subjects.extend([subject] * len(trials))
                all_trial_labels.extend(labels.tolist())
                recording_metadata.append({"subject": subject, "split": split, **meta,
                                           "sha256": _sha256(path), "bytes": path.stat().st_size})
                print(f"Prepared subject {subject} {split}: {len(trials)} trials, {len(names)} channels", flush=True)

        if channel_names is None:
            raise RuntimeError("No recordings were prepared.")
        trials = np.concatenate(all_trials, axis=0)
        model = NeuroRVQTokenizer(n_chans=len(channel_names), n_times=_N_SAMPLES,
                                  sfreq=_SAMPLE_RATE, channel_names=channel_names)
        model.load_pretrained_weights(str(checkpoint_path)).to("cuda").eval()
        log["checkpoint_sha256"] = _sha256(checkpoint_path)
        log["channel_names"] = channel_names
        log["recordings"] = recording_metadata
        outputs = _evaluate(model, trials, torch.device("cuda"), len(channel_names))
        raw = outputs["raw_mse_by_trial"]
        subjects = np.asarray(all_subjects)
        by_subject: dict[str, dict[str, float | int]] = {}
        for subject in _SUBJECTS:
            mask = subjects == subject
            by_subject[str(subject)] = {
                "trial_count": int(mask.sum()),
                "raw_mse": float(raw[mask].mean()),
                "raw_mse_sd": float(raw[mask].std(ddof=1)) if mask.sum() > 1 else 0.0,
            }
        result = {
            "protocol_name": "EEG-Benchmarking preprocessing on all available HGD subjects",
            "trial_count": int(len(raw)),
            "subject_count": len(_SUBJECTS),
            "published_table10_hgd_raw_mse": _TARGET_MSE,
            "pooled_trial_raw_mse": float(raw.mean()),
            "pooled_trial_raw_mse_sd": float(raw.std(ddof=1)),
            "subject_macro_raw_mse": float(np.mean([v["raw_mse"] for v in by_subject.values()])),
            "per_subject": by_subject,
            "band_mse_pooled_trial": {name: float(values.mean())
                                      for name, values in outputs["band_mse_by_trial"].items()},
            "band_mse_subject_macro": {
                name: float(np.mean([values[subjects == subject].mean()
                                     for subject in _SUBJECTS]))
                for name, values in outputs["band_mse_by_trial"].items()},
        }
        log["result"] = result
        log["status"] = "complete"
    except Exception as exc:  # record a machine-readable failure artifact on the runner
        log["status"] = "failed"
        log["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log["elapsed_seconds"] = round(time.time() - started, 3)
        log["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "experiment_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "files": ["experiment_log.json", "artifacts_manifest.json",
                      "metrics.json", "metrics.jsonl", "per_subject.csv"],
            "raw_eeg_exported": False,
            "checkpoint_exported": False,
        }
        (output / "artifacts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if "result" in log:
            (output / "metrics.json").write_text(json.dumps(log["result"], indent=2) + "\n", encoding="utf-8")
            (output / "metrics.jsonl").write_text(
                json.dumps({"run_id": run_id, **log["result"]}) + "\n", encoding="utf-8"
            )
            with (output / "per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("subject", "trial_count", "raw_mse", "raw_mse_sd"))
                writer.writeheader()
                for subject, result in log["result"]["per_subject"].items():
                    writer.writerow({"subject": subject, **result})
        print(json.dumps(log, indent=2), flush=True)


if __name__ == "__main__":
    main()
