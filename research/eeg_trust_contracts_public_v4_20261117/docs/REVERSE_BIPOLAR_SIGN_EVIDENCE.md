# An exact bipolar reversal that geometry alone cannot recover

Source: [Braindecode Draft #1577](https://github.com/braindecode/braindecode/pull/1577), related to [#995](https://github.com/braindecode/braindecode/issues/995).

Under faithful bipolar channel names, `F3-C3 = -(C3-F3)`. Both derivations share the **same spatial midpoint**; treating a midpoint match as an identity copy (+1) is scientifically wrong even though model tensor dimensions and positions agree.

A signed `-1` map is exact for the same two electrode potentials and timebase. The check does **not** infer references across acquisitions, verify montage labeling, claim electrode positions alone capture voltage derivation, or establish real EEG model accuracy.

## Reproduce (CPU, synthetic only)

```bash
python -m pip install numpy mne
python examples/reproduce_reverse_bipolar_mne.py
```

Under seed `20261118`, 256 samples at 256 Hz:
- MNE `F3-C3` vs `-(C3-F3)`: max-abs error `0.0 V`.
- Incorrect +1 reversal: max-abs error `1.0201713701095462e-4 V`.

The upstream change stays in the **existing** `ChannelLayer` and keeps exact-name precedence; it does not add a provenance module or change the project's model APIs. Upstream PR acceptance and CI results must be checked at the exact latest commit.
