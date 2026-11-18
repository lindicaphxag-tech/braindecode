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


## Independent MNE 18-lead reference oracle (new validation)

A second, independent **MNE 1.11.0** EEG integration experiment was executed locally on 18 longitudinal bipolar derivations, five independently seeded synthetic EEG references, five redundant/reverse-polarity trials and two explicit mixed-case named-reference tests. The same 12-case test suite gives **10 passed / 2 failed** on #1593's current-master source-equivalent CPU functions, versus **12/12 passed** on #1597 latest current-head source-equivalent functions (PyTorch 2.10 CPU).

[Public reproducible full-original-Git-blob MNE test](./test_mne_18lead_1597_exact_blob.py) obtains the exact official `channels.py` file with `git show` and checks its Git Blob SHA-1 before executing. This *public original-blob script itself has not run in GitHub Actions yet*; the measured 10/12 and 12/12 results are from local executable functions independently matched to the GitHub bodies. [Latest-head official full-test CI still failed](https://github.com/braindecode/braindecode/actions/runs/44363044210) due a cancelled pytest-coverage step without retrievable traceback.

The test initially contained a **wrong mathematical assertion** that the 18-by-18 reconstruction weights should equal `-I` entrywise; that failed because chained bipolar derivations are linearly dependent, making the minimum-norm solution nonunique. It was replaced with the correct identifiable `W @ R == T` and `W @ (R @ v) == T @ v` invariants. This was a test-design mistake, **not** a production-code failure. All five corrected reverse-polarity seeds pass. MNE's `set_bipolar_reference` provides the external signal-level comparison for the five reference trials. The two actual baseline failures concern mixed-case explicit electrode columns with named references, and are fixed by #1597.

## Current-head authoritative status (2026-11-19)

This section supersedes earlier in-progress CI wording above. The exact head is `baf8e4c734181727a34a33df126a89a356f0cc81`; upstream PR #1597 is **Open and mergeable**, **not CI-green**. [Official workflow #44363044210](https://github.com/braindecode/braindecode/actions/runs/44363044210) completed **failure** on its initial attempt; the full-test job `135009306115` concluded failure and the step `Run Full Test Suite with Coverage` is reported **cancelled**. Five other check runs succeeded. A cancelled test step is *not* evidence of a failed test assertion, nor proof that the failure was environmental.

The tested source-equivalent candidate passed **31/31** numeric/interface regressions, **12/12** MNE integration checks, and **14/14** status-gating unit tests locally. These three local scopes are not interchangeable with official full-suite success. The upstream #1593 baseline **already passes** the empty-target cases; its three observed failures in the 31-case suite concern the explicit electrode-name case contract. The only justified merge gate is an authorized rerun of the failed job or actionable diagnostics on the exact unchanged head. Avoid duplicate review comments and unnecessary production commits solely to retrigger the workflow.

[PR body](https://github.com/braindecode/braindecode/pull/1597) now reflects this exact same status.

## Exact-head evidence refresh: 876f4ec5 (19 Nov 2026)

**Current PR code**: `876f4ec51440b8f7d9b8858cdeef38091198f13c`. The complete `braindecode/functional/channels.py` Git blob is `d056722a3a8381a1cf7f9dec5dafb0fdfcf01db` (independently returned by the GitHub file endpoint for that precise ref). The public [31-case source loader](./test_derivation_1597_post1593_exact_blob.py) and [CPU workflow](../../.github/workflows/derivation-1597-post1593-exact-blob-cpu.yml) were updated to fetch that exact commit and verify its full raw Git blob ID. The old #1593 baseline blob `896f80ac2934851a3ef7ef731c83ac43be0aa845` remains fixed for comparison.

The production patch is **now only the explicit electrode-column normalization**; the empty-target early return was removed as redundant on current master. The PR has three changed files (+35/-1), with regression tests still asserting both empty-target return shapes and adding the named-reference case.

**Official current-head CI**: [new tests workflow #44380976465](https://github.com/braindecode/braindecode/actions/runs/44380976465) was **in_progress** at this verification. Acceptance, docs, pre-commit and whats-new checks completed successfully in the observed runs. Do not apply the historical `baf8e4c7` CI failure to the new head, or claim that the updated public full-Git-blob workflow has executed unless a successful run is independently observed.

**Evidence boundary**: 28/31 versus 31/31, MNE 10/12 versus 12/12, and the 14/14 truth-gate results predate this exact HEAD. The tests are now correctly pinned for a new execution; updating expected hashes alone does **not** constitute a fresh passing run.

## Exact-head official success verification (2026-11-19)

**Braindecode #1597**, HEAD `876f4ec51440b8f7d9b8858cdeef38091198f13c`: the official [full tests workflow #44380976465](https://github.com/braindecode/braindecode/actions/runs/44380976465) has **completed/success**. GitHub's exact-commit Check Runs endpoint returned **7/7 completed/success**, namely `test (ubuntu-latest, 3.12)`, `acceptance (ubuntu-latest, 3.12)`, `codecov/project`, `Codacy Static Code Analysis`, `build_docs`, `pre-commit`, and `check-whats-news`. Both jobs in the tests workflow succeeded. This is genuine official upstream current-head CI, not inherited evidence from old SHAs. The old `baf8e4c7` full-suite failure is superseded for current merge-readiness.

**State at verification**: PR **Open**, **mergeable=true**, **not yet merged**; no approval is implied solely from successful checks. Current diff remains three files +35/-1, with one production change for explicit-electrode case normalization, plus regression tests and release notes. The publicly pinned source/blob CPU script is updated, but its fork-hosted Actions workflow still has **zero observed runs**. Historical local 31/31 and MNE 12/12 remain separate evidence and were not reexecuted on this HEAD.

**Next gate**: maintainer review/merge. Avoid duplicate reviewer pings, branch churn, or redundant CI runs without reviewer request.

## Formal upstream merge — final status (2026-11-19)

The follow-up **Braindecode #1597 is formally merged**: https://github.com/braindecode/braindecode/pull/1597. GitHub reports `merged_at=2026-11-18T20:26:44Z` (2026-11-19 04:26:44 Singapore/Beijing), merge commit **`41fc193cece2b20af9d89912dfdf0c036eac8f56`**, accepted PR head `876f4ec51440b8f7d9b8858cdeef38091198f13c`. The seven official exact-head checks, including full unit tests, completed **success** before merge. Earlier `Open`, `pending`, `failed`, or `in_progress` statements in this historical log describe earlier checkpoints and are superseded as of this final status.

The public fork's exact-blob CPU workflow still had no run in the last separate fork query; its existence is a reproducible artifact, not claimed as an additional cloud success. Stable release adoption and final release metadata remain independently unverified; the latest published Braindecode GitHub release was v2.0.0 and v2.1.0 release-preparation PR #1571 remained Open/Draft.
