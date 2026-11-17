# Cross-dataset identity lineage preflight v2

This optional extension addresses an important limit of subject-held-out evaluation: dataset-local subject identifiers cannot establish independent humans across datasets. A recording may also be repackaged under a new local recording ID.

## New optional declared fields

`AuditRecord.global_subject_id` — an explicitly curated stable subject key across dataset boundaries, where lawful to use.

`AuditRecord.origin_recording_id` — an explicitly curated stable origin recording key across datasets or derivative exports.

Both remain optional to preserve older callers. Never infer cross-dataset equivalence from matching local subject strings. No patient identification, biometric matching or raw EEG content hashing is performed by this helper.

## New detection and coverage outputs

- `global_subject_overlap` and `origin_recording_overlap` invalidate the train/validation/test split when known keys collide.
- `identity_coverage` reports the number of records supplying each key by partition.
- `cross_dataset_identity_status=PARTIAL_OR_UNAVAILABLE` when any record lacks either key.
- `CHECKED_DECLARED_GLOBAL_IDS` indicates only that declared keys were compared, **not** that they are verified or that unknown shared ancestry is absent.
- Existing dataset-local group disjointness, missing-partition and pretraining-corpus caveats are preserved.

## Scientific caution

Even complete key coverage cannot certify held-out performance without independent verification of the key mapping and upstream pretraining subjects. Cross-dataset leakage absence is a stronger claim than merely checking named dataset overlap. No real-data results or EEG decoding gains are asserted here.

## Reproduction

`PYTHONPATH=research/eeg_trust_contracts_20261117/src python -m pytest -q research/eeg_trust_contracts_20261117/tests/test_evaluation_audit.py`

Source: Braindecode [#1492](https://github.com/braindecode/braindecode/issues/1492), merged [#1493](https://github.com/braindecode/braindecode/pull/1493) and MOABB cross-subject evaluation; this is additional research tooling, not an official MOABB or Braindecode API.
