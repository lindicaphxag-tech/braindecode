"""EEG trust research prototype: dependency-light signal and evaluation tools.

PyTorch is only imported when the optional pretrained-model helper is accessed.
"""
from .signal_semantics import Channel, Manifest, classify, apply_transform
from .evaluation_audit import AuditRecord, CorpusStatement, audit_evaluation

__all__ = [
    "Channel", "Manifest", "classify", "apply_transform",
    "AuditRecord", "CorpusStatement", "audit_evaluation", "audit_model",
]


def __getattr__(name):
    if name == "audit_model":
        from .pretrained_audit import audit_model
        return audit_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
