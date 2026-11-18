# Braindecode #1597 after maintainer #1593 — public reviewer evidence

**This document and all linked test source are public, readable without contributor access.** The corresponding upstream [Braindecode PR #1597](https://github.com/braindecode/braindecode/pull/1597) is currently Open and mergeable, not yet accepted.

## Exact source commits

| Code stage | Git commit | Original `braindecode/functional/channels.py` Git blob |
|---|---|---|
| Maintainer #1593 base | `9c541453e2c13ce0c3680a1582a136ca121b0da7` | `896f80ac2934851a3ef7ef731c83ac43be0aa845` |
| Current #1597 | `baf8e4c734181727a34a33df126a89a356f0cc81` | `3b05938f993e7fd6f1e6651614018129dbb8defc` |

The latest #1597 commit merged upstream master #1593 without force-push; its parents are `0f619162` and `9c541453`. GitHub reports `mergeable=true` with only **3 files (+36/−1)** changed from the current upstream base. It preserves #1593's public `linear_channel_map` and the new named `reference` argument.

## Native local PyTorch CPU red/green

| Executable version | Actual result (31 cases) | Characterization |
|---|---|---|
| After #1593, before #1597 | **28 passed / 3 failed** | The three failures are all case-normalization of explicitly supplied electrodes, including the #1593 `reference='Cz'` API |
| Current #1597 merge commit | **31/31 passed** | All of the above fixed; preexisting derivation, gradient, and reference-invariance controls remain green |

Test environment: Python 3.13, PyTorch `2.10.0+cpu`, pytest `9.0.2`, CPU only. Cases include 2 empty-target variants, 2 explicit column-case variants, one new named-reference case, 9 preexisting exact-derivation contracts, partially identifiable rows, gradient sign, 12 seeded electrode-potential identities, mixed-case channel labels and nonidentifiable-row zeroing.

The two locally executed modules consist of the actual upstream function bodies (`derivation_matrix`, `linear_channel_map`, `exact_derivation`) obtained from GitHub, with docstrings and the unrelated spherical-spline routine omitted. For each stage, the whitespace-normalized function content had the **same character length and FNV-1a64 fingerprint** independently computed from the corresponding official GitHub source:

| Stage | Normalized characters | FNV-1a64 fingerprint |
|---|---:|---|
| #1593 | `2199` | `d67280dd781eafe7` |
| #1597 | `2314` | `6c6724661252d0e5` |

These fingerprints verify code-body equivalence in this local audit but **are not cryptographic identity of the entire Git source**. For byte-exact original source, use the [public full-Git-blob pinned pytest script](./test_derivation_1597_post1593_exact_blob.py), which runs `git show` and validates the raw Git Blob SHA-1 before executing the whole original module.

**A substantive change in interpretation:** On the #1593 base, the empty-target cases already pass. Accordingly, the explicit early return is now a fast, well-defined exit path, not an active empty-target crash in current `master`. The 27-pass/3-fail → 30-pass old evidence in [the previous report](./RESULTS_1597_RED_GREEN.md) remains correct for the *earlier* #1577 source, not for the new #1593 baseline.

## Independent full-blob command (runs without private-hosted Actions)

```bash
# In a local full clone of lindicaphxag-tech/braindecode:
git fetch origin fix/exact-derivation-empty-and-case-20261119
git fetch origin evidence/1597-exact-head-redgreen-20261119
git checkout evidence/1597-exact-head-redgreen-20261119
python -m pip install torch pytest numpy
BD_CHANNELS_REF=merged python -m pytest -q research/validation/test_derivation_1597_post1593_exact_blob.py
# expected 28 passed, 3 failed (baseline)
BD_CHANNELS_REF=followup python -m pytest -q research/validation/test_derivation_1597_post1593_exact_blob.py
# expected 31 passed (candidate)
```

The [public CPU-only GitHub Actions recipe](../../.github/workflows/derivation-1597-post1593-exact-blob-cpu.yml) automates that script, but the fork had **not generated a run** when last observed. Do not treat it as a successful CI result.

On the **official latest #1597 head**, five GitHub check runs (docs, acceptance, pre-commit, static analysis, whats-new) had succeeded; the large full-test job was not yet final at last check. [Official latest-head CI run](https://github.com/braindecode/braindecode/actions/runs/44363044210). No accepted Merge or stable release inclusion is claimed.


## Subsequent official full-test result (19 Nov audit)

**Exact HEAD** `baf8e4c734181727a34a33df126a89a356f0cc81` ultimately produced **5/6 GitHub checks successful, 1 failure**. The failed `test (ubuntu-latest, 3.12)` job's step `Run Full Test Suite with Coverage` was reported **cancelled**, and the full GitHub Actions run completed **failure** at 2026-11-18 18:43:15 UTC. [Exact run](https://github.com/braindecode/braindecode/actions/runs/44363044210). The earlier five checks (acceptance, docs, pre-commit, changelog, static) remain green. The previous head `0f619162`'s successful 7/7 status is **historical** and not valid for the current SHA.

A matching `master` `tests` workflow [succeeded on upstream #1593](https://github.com/braindecode/braindecode/actions/runs/44351027114), and another [contemporaneous PR's tests succeeded](https://github.com/braindecode/braindecode/actions/runs/44358021240). The repository's [`tests.yml`](https://github.com/braindecode/braindecode/blob/baf8e4c734181727a34a33df126a89a356f0cc81/.github/workflows/tests.yml) has a 60-minute job timeout and parallel `pytest -n auto` with coverage; the observed failure occurred after around 53 minutes. A cancelled step does **not** itself prove either a test assertion failure or a timeout, and GitHub's job-log fetch currently returned 404 BlobNotFound. [Existing upstream reviewer evidence comment](https://github.com/braindecode/braindecode/pull/1597#issuecomment-6987352048) was **edited in place** with this exact-head update; a single-job rerun attempted via the connected GitHub integration returned permission error 403. No fresh run was triggered.

**Local verification remains 31/31 passing** for the exact-function-body equivalent sample. The public full-Git-blob workflow in this branch has not produced its own new Actions run. Until the upstream full-test run succeeds or a maintainer identifies the first failing log, #1597 is **Open, mergeable but not CI-ready**. 
