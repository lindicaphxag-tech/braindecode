# Authors: Braindecode contributors.
# License: BSD-3-Clause
"""Leakage-safe subject/session cross-validation on recording datasets."""

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset

from braindecode.datasets import (
    BaseConcatDataset,
    create_grouped_folds,
    validate_group_disjointness,
)


class SmallRecording(Dataset):
    def __init__(self, subject, session, record_id, n_windows=6):
        self.description = pd.Series(
            {"subject": subject, "session": session, "record_id": record_id}
        )
        self.n_windows = n_windows

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        return np.array([idx]), idx % 2


@pytest.fixture
def dataset():
    recordings = [
        SmallRecording(subject, session, f"{subject}-{session}")
        for subject in range(4)
        for session in ("A", "B")
    ]
    transform = lambda target: target + 1
    return BaseConcatDataset(recordings, target_transform=transform)


def _recording_ids(partition):
    return set(partition.description["record_id"])


@pytest.mark.parametrize("n_splits", [2, 4])
def test_subject_folds_disjoint_complete_and_recording_atomic(dataset, n_splits):
    folds = create_grouped_folds(dataset, group_by="subject", n_splits=n_splits)
    assert len(folds) == n_splits
    all_recordings = _recording_ids(dataset)
    test_sets = []
    for fold in folds:
        validate_group_disjointness(fold["train"], fold["test"], "subject")
        assert fold["train"].target_transform is dataset.target_transform
        assert fold["test"].target_transform is dataset.target_transform
        test_sets.append(_recording_ids(fold["test"]))
        assert _recording_ids(fold["train"]) | test_sets[-1] == all_recordings
        assert not (_recording_ids(fold["train"]) & test_sets[-1])
        assert len(fold["train"]) == len(fold["train"].datasets) * 6
        assert len(fold["test"]) == len(fold["test"].datasets) * 6
    assert set.union(*test_sets) == all_recordings
    assert sum(map(len, test_sets)) == len(all_recordings)


def test_group_shuffle_is_reproducible(dataset):
    def assignments(seed):
        folds = create_grouped_folds(
            dataset, "subject", 4, shuffle=True, random_state=seed
        )
        return tuple(tuple(sorted(_recording_ids(f["test"]))) for f in folds)

    assert assignments(31) == assignments(31)
    assert assignments(31) != assignments(19)


def test_composite_subject_session_groups_are_supported(dataset):
    folds = create_grouped_folds(dataset, ["subject", "session"], n_splits=4)
    for fold in folds:
        validate_group_disjointness(fold["train"], fold["test"], ["subject", "session"])
    union = set().union(*(_recording_ids(f["test"]) for f in folds))
    assert union == _recording_ids(dataset)


def test_subject_validator_detects_leakage_even_when_session_differs(dataset):
    train = dataset.split({"train": [0, 2]})["train"]
    test = dataset.split({"test": [1, 3]})["test"]
    with pytest.raises(ValueError, match="leakage"):
        validate_group_disjointness(train, test, "subject")
    validate_group_disjointness(train, test, ["subject", "session"])


def test_missing_group_column_fails_closed(dataset):
    with pytest.raises(ValueError, match="missing"):
        create_grouped_folds(dataset, "participant_id", n_splits=2)


def test_missing_group_identity_fails_closed(dataset):
    dataset.datasets[0].description["subject"] = None
    with pytest.raises(ValueError, match="missing"):
        create_grouped_folds(dataset, "subject", n_splits=2)


@pytest.mark.parametrize("invalid", [0, 1, -2, True, 5])
def test_invalid_fold_count_does_not_silently_leak(dataset, invalid):
    with pytest.raises(ValueError, match="n_splits"):
        create_grouped_folds(dataset, "subject", n_splits=invalid)


@pytest.mark.parametrize("invalid", [[], [""], ["subject", "subject"]])
def test_invalid_group_by_names_fail_closed(dataset, invalid):
    with pytest.raises(ValueError):
        create_grouped_folds(dataset, invalid, n_splits=2)


def test_shuffle_seed_must_be_explicitly_enabled(dataset):
    with pytest.raises(ValueError, match="shuffle"):
        create_grouped_folds(dataset, "subject", n_splits=2, random_state=123)


def test_grouped_folds_is_not_window_level_split(dataset):
    first = create_grouped_folds(dataset, "subject", n_splits=2)
    second = create_grouped_folds(dataset, "subject", n_splits=2)
    assert [
        _recording_ids(f["test"]) for f in first
    ] == [_recording_ids(f["test"]) for f in second]
    # Recording objects remain intact; no window-level subsets are created.
    assert all(
        isinstance(rec, SmallRecording)
        for f in first
        for rec in f["test"].datasets
    )
