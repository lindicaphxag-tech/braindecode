"""Run conservative manifest audits locally without downloading EEG."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
from .signal_semantics import Channel, Manifest, classify
from .evaluation_audit import AuditRecord, CorpusStatement, audit_evaluation


def manifest_from_dict(obj):
    channels = []
    for ch in obj["channels"]:
        terms = ch.get("terms")
        channels.append(Channel(ch["name"], None if terms is None
                                else tuple((node, Fraction(str(weight))) for node, weight in terms.items())))
    return Manifest(tuple(channels), obj.get("acquisition_id"), obj.get("timebase_id"),
                    obj.get("sampling_hz"), obj.get("unit"))


def main():
    ap = argparse.ArgumentParser(description="Offline EEG signal and evaluation metadata preflight")
    ap.add_argument("mode", choices=("signal", "evaluation"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.mode == "signal":
        result = classify(manifest_from_dict(data["source"]), manifest_from_dict(data["target"]))
    else:
        rows = [AuditRecord(**r) for r in data["records"]]
        pre = data["pretraining"]
        result = audit_evaluation(rows, CorpusStatement(frozenset(pre["datasets"]),
                                  bool(pre["complete"]), pre.get("source_reference")),
                                  require_site_holdout=data.get("require_site_holdout", False))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
