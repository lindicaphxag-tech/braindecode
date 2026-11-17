"""A minimal semantics-only installation must not import PyTorch eagerly."""
import os
import subprocess
import sys
from pathlib import Path


def test_semantics_available_with_torch_explicitly_blocked():
    script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('simulated missing torch')
sys.meta_path.insert(0, BlockTorch())
import trust_contract
from trust_contract import Channel, Manifest, classify
from trust_contract.cli import manifest_from_dict
from trust_contract.evaluation_audit import AuditRecord, CorpusStatement, audit_evaluation
assert 'torch' not in sys.modules
print('semantics-and-evaluation-imports-work-without-torch')
"""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo / 'src')
    res = subprocess.run([sys.executable, '-c', script], env=env, text=True,
                         capture_output=True, check=True)
    assert 'without-torch' in res.stdout


def test_optional_pretrained_helper_is_resolved_on_access():
    from trust_contract import audit_model
    assert callable(audit_model)
