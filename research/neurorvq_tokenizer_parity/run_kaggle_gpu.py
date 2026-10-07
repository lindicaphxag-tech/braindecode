"""Bootstrap the pinned NeuroRVQ benchmark run on a Kaggle GPU kernel."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _install(*requirements: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *requirements],
        check=True,
    )


def main() -> None:
    work = Path("/kaggle/working")
    source = Path("/kaggle/temp/braindecode-neurorvq-evidence")
    os.environ.setdefault("OUTPUT_DIR", str(work / "neurorvq_table10"))
    os.environ.setdefault("HGD_DATA_DIR", "/kaggle/temp/high_gamma_edf")
    os.environ.setdefault("HF_HOME", "/kaggle/temp/hf_cache")
    _install("mne", "edfio", "huggingface_hub")
    _install(
        "git+https://github.com/lindicaphxag-tech/braindecode.git"
        "@61194d513e0c5234a2f55923bf157643354e4b42"
    )
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "research/neurorvq-tokenizer-parity",
         "https://github.com/lindicaphxag-tech/braindecode.git", str(source)],
        check=True,
    )
    replication = source / "research" / "neurorvq_tokenizer_parity" / "reproduce_table10_highgamma.py"
    subprocess.run([sys.executable, str(replication)], check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
