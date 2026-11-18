# Independent exact-source red/green evidence for Braindecode #1597

**PR focus:** [Braindecode #1597](https://github.com/braindecode/braindecode/pull/1597) follows the officially merged [#1577](https://github.com/braindecode/braindecode/pull/1577). It corrects empty-target reductions and case-insensitive explicitly supplied electrode columns; it does not change the underlying row-span/pseudoinverse design.

**Reproducible script:** [`research/validation/test_derivation_1597_redgreen.py`](./test_derivation_1597_redgreen.py) in this branch. The script uses `git show <exact commit>:braindecode/functional/channels.py` from the author's public Braindecode fork and verifies the exact Git Blob SHA-1 before importing; this avoids manually transcribed sources and local-package drift.

| Stage | Commit | Expected Git source blob | CPU results |
|---|---|---|---|
| Prior merged #1577 source | `0bdad6ae0109133d89e3be6a002592a3670108a2` | `428b914fa6392453a10673436db2fa4c0fdc5510` | 27 passed / 3 failed (expected) |
| #1597 proposed source | `0f61916201c573b7d27dd2c1bdc8538f1850a206` | `4c3ca115b3ec73e4ed6aabded0400f510c3bcab5` | 30 passed / 0 failed |

**Local platform:** Python 3, PyTorch `2.10.0+cpu`, pytest `9.0.2`, standard CPU. The separate **30-case local suite** used both exact-blob verified implementations and covers: two empty-target shapes, two explicit electrode-name case contracts, nine preexisting exactness cases, partial identifiability, signed gradient propagation, 12 fixed-seed random electrode-potential reconstructions, mixed-case input names, nonidentifiable-row zeroing, and blob identity.

Failure detail: old `exact_derivation([],[])` raises `IndexError: amax(): Expected reduction dim 1 to have non-zero size`; uppercase explicit columns produced `['F3','C3','f3','c3']` instead of `['f3','c3']`, and mixed-case duplicates remained separate. The old nonempty-input empty-target control passes; do not overclaim four old failures.

## Run the public proof code

```bash
python -m pip install pytest torch
BD_CHANNELS_REF=merged pytest -q research/validation/test_derivation_1597_redgreen.py
# expected nonzero exit (27 passed, 3 failed)
BD_CHANNELS_REF=followup pytest -q research/validation/test_derivation_1597_redgreen.py
# expected zero exit (30 passed)
```

The upstream fork evidence branch is based on the actual head `0f619162`; official `get_pr_info` on the upstream PR and Git `compare_commits` confirm that merged #1577 source commit is its ancestor. This branch is **not** an official PR and should not be counted as new external merged work.

**Validation boundary:** the 30/30 and 27/30 runs were conducted locally using byte-identical source files, not the full Braindecode package or upstream CI. The public script is supplied to independently reproduce the same cases in a clean checkout. Official CI and maintainers decide acceptance.
