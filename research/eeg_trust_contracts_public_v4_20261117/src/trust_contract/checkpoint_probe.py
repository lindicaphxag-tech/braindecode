"""Opt-in, pinned-checkpoint provenance probe for Braindecode (research tool).

No downloads happen unless --allow-download is explicitly supplied alongside
a 40-character Hub commit hash. This checks tensor origin metadata, not model
benchmark performance or fidelity to a released reference implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .pretrained_audit import audit_model

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


def _source(args) -> tuple[Path, str | None]:
    if args.snapshot is not None:
        if args.allow_download or args.repo_id or args.revision:
            raise ValueError("a local snapshot cannot be combined with Hub download arguments")
        path = Path(args.snapshot)
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("snapshot must be a directory containing config.json")
        return path, None
    if not args.allow_download:
        raise ValueError("network download requires explicit --allow-download")
    if not args.repo_id or not args.revision or not _COMMIT.fullmatch(args.revision):
        raise ValueError("Hub download requires --repo-id and pinned 40-hex --revision")
    from huggingface_hub import snapshot_download

    directory = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision.lower(),
        allow_patterns=["config.json", "model.safetensors", "*.safetensors", "README.md"],
    )
    return Path(directory), args.revision.lower()


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def run_probe(args, *, model_classes: Any = None) -> dict:
    """Load one pinned/local model and inspect its checkpoint origin metadata."""
    source, pinned = _source(args)
    if model_classes is None:
        import braindecode.models as model_classes
    cls = getattr(model_classes, args.model_class, None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise ValueError("--model-class must name a Braindecode model with from_pretrained")
    kwargs = {}
    if args.kwargs_json:
        kwargs = json.loads(Path(args.kwargs_json).read_text(encoding="utf-8"))
        if not isinstance(kwargs, dict):
            raise ValueError("--kwargs-json must contain a JSON object")
    model = cls.from_pretrained(source, local_files_only=True, **kwargs)
    if args.freeze:
        model.requires_grad_(False)
    report = audit_model(
        model,
        expected_new=args.expected_new,
        feature_scope=args.feature_scope if args.feature_scope is not None else None,
        include_buffers=args.include_buffers,
    )
    import torch
    file_hashes = {
        path.name: _sha256(path) for path in sorted(source.glob("*.safetensors"))
    }
    report.update({
        "probe_version": "0.1",
        "source_kind": "hub_pinned" if pinned else "local_snapshot",
        "requested_hub_commit": pinned,
        "model_class_requested": args.model_class,
        "torch_version": torch.__version__,
        "safetensors_sha256": file_hashes,
        "feature_scope_is_manually_declared": True,
        "not_proven": [
            "forward reference fidelity", "EEG task performance",
            "checkpoint trustworthiness", "channel or reference semantics",
            "pretraining subject independence",
        ],
    })
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path, help="Existing local checkpoint directory")
    group.add_argument("--repo-id", help="Hub model repo; requires explicit download and pinned revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--revision", help="Exact 40-character Hugging Face commit hash")
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--kwargs-json", type=Path)
    parser.add_argument("--feature-scope", action="append", default=None)
    parser.add_argument("--expected-new", action="append", default=[])
    parser.add_argument("--include-buffers", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    result = run_probe(args)
    data = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
