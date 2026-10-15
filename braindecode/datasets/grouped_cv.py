# Authors: Braindecode contributors.
# License: BSD-3-Clause
"""Leakage-safe cross-validation at the recording metadata level."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .base import BaseConcatDataset


def _group_codes(
    dataset: BaseConcatDataset, group_by: str | Sequence[str]
) -> np.ndarray:
    if not isinstance(dataset, BaseConcatDataset):
        raise TypeError("Expected a BaseConcatDataset with recording descriptions.")
    columns = [group_by] if isinstance(group_by, str) else list(group_by)
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError("group_by must specify at least one nonempty metadata column.")
    if len(set(columns)) != len(columns):
        raise ValueError("group_by cannot contain duplicate column names.")
    description = dataset.description
    missing_columns = [
        column for column in columns if column not in description.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Grouping columns missing from dataset.description: {missing_columns}"
        )
    if description[columns].isna().any().any():
        raise ValueError(
            "Grouping metadata cannot contain missing subject/session identities."
        )

    # pandas factorization maps arbitrary hashable composite metadata keys
    # to dense group IDs. Split indices remain recording-level, never windows.
    index = pd.MultiIndex.from_frame(description[columns])
    codes, _ = pd.factorize(index, sort=False)
    return codes.astype(np.int64, copy=False)


def validate_group_disjointness(
    train: BaseConcatDataset,
    test: BaseConcatDataset,
    group_by: str | Sequence[str] = "subject",
) -> None:
    """Raise when a subject/session group occurs in both partitions.

    Checks original description values, not the independently factorized
    integer group IDs of each subset.
    """
    columns = [group_by] if isinstance(group_by, str) else list(group_by)
    _group_codes(train, columns)
    _group_codes(test, columns)
    train_keys = set(train.description[columns].itertuples(index=False, name=None))
    test_keys = set(test.description[columns].itertuples(index=False, name=None))
    overlap = train_keys.intersection(test_keys)
    if overlap:
        raise ValueError(
            "Group leakage detected across training and testing "
            f"({len(overlap)} shared groups)."
        )


def create_grouped_folds(
    dataset: BaseConcatDataset,
    group_by: str | Sequence[str] = "subject",
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[dict[str, BaseConcatDataset]]:
    """Create disjoint train/test grouped cross-validation folds.

    Each test recording appears in exactly one fold and each group is wholly
    assigned to one partition. With shuffle=False, GroupKFold balances the
    number of recordings per fold. With shuffle=True, KFold distributes
    randomly permuted *groups*, preserving group purity and reproducibility.

    Note that this deliberately does not stratify target labels; doing so
    requires a recording-level target definition that windowed datasets may
    not provide.
    """
    groups = _group_codes(dataset, group_by)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    group_ids = np.unique(groups)
    if n_splits > len(group_ids):
        raise ValueError(
            f"n_splits={n_splits} exceeds the {len(group_ids)} independent groups."
        )
    if random_state is not None and not shuffle:
        raise ValueError("random_state can only be specified when shuffle=True.")

    if shuffle:
        # Shuffle units of independence, not windows or recordings.
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = (
            (
                np.flatnonzero(~np.isin(groups, group_ids[test_groups])),
                np.flatnonzero(np.isin(groups, group_ids[test_groups])),
            )
            for _, test_groups in cv.split(group_ids)
        )
    else:
        cv = GroupKFold(n_splits=n_splits)
        iterator = cv.split(np.zeros(len(groups)), groups=groups)

    folds = []
    coverage = np.zeros(len(groups), dtype=np.int64)
    for train_indices, test_indices in iterator:
        coverage[test_indices] += 1
        fold = dataset.split(
            {"train": train_indices.tolist(), "test": test_indices.tolist()}
        )
        validate_group_disjointness(fold["train"], fold["test"], group_by=group_by)
        folds.append(fold)

    if not np.all(coverage == 1):
        raise RuntimeError(
            "Each recording must occur exactly once among all test folds."
        )
    return folds
