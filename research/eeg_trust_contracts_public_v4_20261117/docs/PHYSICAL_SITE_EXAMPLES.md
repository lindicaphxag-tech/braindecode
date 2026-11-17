# Three executable site-holdout counterexamples

Run from the repository root using the existing metadata CLI:

```bash
python -m pip install -e './research/eeg_trust_contracts_20261117[tests]'
for f in research/eeg_trust_contracts_20261117/examples/site_holdout_*.json; do
  python -m trust_contract.cli evaluation --input "$f"
done
```

| Input | Required result | Why |
|---|---|---|
| `site_holdout_unknown.json` | `split_valid=false`, `site_holdout_unverified` | Distinct dataset-local site labels cannot establish physical site independence |
| `site_holdout_same_physical_site.json` | `split_valid=false`, `global_site_overlap` | The hospital has different dataset-local labels but the same declared physical site identity |
| `site_holdout_distinct_declared_sites.json` | `split_valid=true`, `CHECKED_DECLARED_SITE_IDS` | Declared distinct keys pass the **metadata preflight only**, not scientific proof of physical independence |

Local CPU reproduction of the corresponding source implementation: **66 passed**, with these CLI scenarios matching the table. No real EEG, hospitals, patients, or actual facility identifiers are in these files.

Security: publicly released site IDs here are fabricated. Users with real multi-center data should pseudonymize stable site keys under an approved governance protocol and must not release sensitive linkage tables.

The three examples target a gap in *site-held-out evaluation claims*; they do not themselves establish a generalizable EEG benchmark result or a new official MOABB API.
