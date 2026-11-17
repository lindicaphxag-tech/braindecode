"""Offline probe tests; no Braindecode import or real weights."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from trust_contract.checkpoint_probe import build_parser, run_probe


class LocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.head = nn.Linear(3, 2)
        self._pretrained_provenance = {
            "loaded": {"encoder.weight", "encoder.bias"},
            "mixed": set(), "repo_id": "mock/source", "commit": "abc123",
        }

    @classmethod
    def from_pretrained(cls, snapshot, **kwargs):
        assert (Path(snapshot) / "config.json").exists()
        assert kwargs == {"local_files_only": True}
        return cls()


def fixture(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic bytes, not actual weights")
    return tmp_path


def test_offline_probe_reports_frozen_new_feature_and_file_hash(tmp_path):
    fixture(tmp_path)
    args = build_parser().parse_args([
        "--snapshot", str(tmp_path), "--model-class", "LocalModel", "--freeze",
        "--feature-scope", "head.", "--expected-new", "head.",
    ])
    report = run_probe(args, model_classes=SimpleNamespace(LocalModel=LocalModel))
    assert report["source_kind"] == "local_snapshot"
    assert report["loaded_fraction"] < 1
    assert report["feature_unloaded_frozen"] == ["head.bias", "head.weight"]
    assert report["unexpected_new_or_adapted"] == []
    assert len(report["safetensors_sha256"]["model.safetensors"]) == 64
    assert report["requested_hub_commit"] is None


def test_hub_requires_pinned_commit_and_explicit_download():
    for argv, error in [
        (["--repo-id", "owner/model"], "allow-download"),
        (["--repo-id", "owner/model", "--allow-download"], "40-hex"),
        (["--repo-id", "owner/model", "--allow-download", "--revision", "main"], "40-hex"),
    ]:
        args = build_parser().parse_args([*argv, "--model-class", "LocalModel"])
        with pytest.raises(ValueError, match=error):
            run_probe(args, model_classes=SimpleNamespace(LocalModel=LocalModel))


def test_nonexistent_snapshot_and_unknown_model_are_rejected(tmp_path):
    args = build_parser().parse_args(["--snapshot", str(tmp_path), "--model-class", "LocalModel"])
    with pytest.raises(ValueError, match="containing config.json"):
        run_probe(args, model_classes=SimpleNamespace(LocalModel=LocalModel))
    fixture(tmp_path)
    with pytest.raises(ValueError, match="Braindecode model"):
        run_probe(args, model_classes=SimpleNamespace())


def test_invalid_json_kwargs_refused(tmp_path):
    fixture(tmp_path)
    kw = tmp_path / "args.json"
    kw.write_text("[]")
    args = build_parser().parse_args([
        "--snapshot", str(tmp_path), "--model-class", "LocalModel",
        "--kwargs-json", str(kw),
    ])
    with pytest.raises(ValueError, match="JSON object"):
        run_probe(args, model_classes=SimpleNamespace(LocalModel=LocalModel))
