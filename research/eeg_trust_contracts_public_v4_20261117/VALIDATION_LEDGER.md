# Validation ledger — independent research branch, 2026-11-17

- Locally executed 68/68 Python CPU tests on synthetic EEG/channel metadata and stub pretrained sources.
- Includes exact rational derivation algebra; one MNE bipolar-reference parity fixture; cross-dataset declared-subject, source-recording and physical-site lineage checks; optional metadata provenance and torch-free import isolation.
- Three CLI site examples: missing global site id => NOT CERTIFIED; aliases share physical id => OVERLAP; distinct declared physical ids => METADATA PREFLIGHT PASS (not verified independent hospitals).
- Pinned real Hub checkpoint workflow exists but has NOT been run against released model weights. Hugging Face Jobs returned 402 Payment Required in this attempt; local environment has no direct network access.
- Earlier GitHub private CI for research package had no runner allocated (empty job steps); do not interpret this as a test failure or assert an official pass.
- Public GitHub branch is a SOURCE MIRROR, not upstream merge. Braindecode #1574 was **closed without merge** following explicit maintainer feedback: merged #1558 already provides the desired per-module pretrained provenance, so a second provenance module is redundant. Keep the standalone prototype as independent research only; do not imply upstream adoption. #995/1227 remain discussions. A narrow existing-module bipolar-polarity draft, Braindecode #1577, is separately under official CI/review.
- GitHub provenance: private PR #86 (physical site preflight), #87 (packaging and lazy import); both merged into personal private research repository. Author: Zhibo Zhang (@lindicaphxag-tech).

**Claims excluded:** no real EEG accuracy gains, clinical validation, pretraining corpus independence certification, fidelity to a released model checkpoint, maintainer privilege, or paper acceptance.
