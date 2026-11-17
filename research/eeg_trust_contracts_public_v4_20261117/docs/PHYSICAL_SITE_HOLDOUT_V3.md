# Physical site lineage across EEG datasets

A different dataset name or local hospital label is not proof of cross-site independence. This extension adds an optional caller-supplied `global_site_id` to the audit record, compared only when `require_site_holdout=True`. It preserves the old non-strict behavior by default.

**Fail-closed preflight:** If different datasets appear across train/validation/test and any of their compared records lack a global physical-site key, the strict site-holdout split is invalidated with `site_holdout_unverified`. Reused keys are flagged as `global_site_overlap`. Coverage per split is reported under `identity_coverage[*].global_site_ids`.

**Validation:** A corresponding local implementation passed 66/66 CPU synthetic tests, including eight newly parameterized physical-site tests. GitHub runner results must be checked separately; this is neither a real EEG cross-site generalization result nor an upstream Braindecode/MOABB API.

**Privacy and provenance:** Do not expose real hospital codes, raw participant identities or linkable patient data publicly. Use approved pseudonymous identifiers. This audit compares **asserted** keys; a false mapping is not detected, and its absence does not certify site independence.

**Scope:** Distinct from named pretraining dataset overlap (#1493). A prospective study requires verified acquisition-site mapping, actual real-data splits, fixed preprocessing, and subject/session independence checks.
