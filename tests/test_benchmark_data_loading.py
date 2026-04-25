from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_default_prediction.data_loading.benchmark import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    load_and_validate_benchmark_dataset,
    split_features_target,
    validate_benchmark_dataset,
)


def _make_valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TARGET_COLUMN: [1, 0],
            "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.1],
            "age": [45, 50],
            "NumberOfTime30-59DaysPastDueNotWorse": [2, 0],
            "DebtRatio": [0.8, 0.2],
            "MonthlyIncome": [9120, 2600],
            "NumberOfOpenCreditLinesAndLoans": [13, 4],
            "NumberOfTimes90DaysLate": [0, 0],
            "NumberRealEstateLoansOrLines": [6, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
            "NumberOfDependents": [2, 1],
        }
    )


def test_validate_benchmark_dataset_accepts_expected_schema() -> None:
    frame = _make_valid_frame()

    validate_benchmark_dataset(frame)


def test_validate_benchmark_dataset_rejects_bad_target_values() -> None:
    frame = _make_valid_frame()
    frame.loc[0, TARGET_COLUMN] = 2

    with pytest.raises(ValueError, match="invalid values"):
        validate_benchmark_dataset(frame)


def test_split_features_target_returns_features_and_target() -> None:
    frame = _make_valid_frame()

    features, target = split_features_target(frame)

    assert TARGET_COLUMN not in features.columns
    assert list(target.tolist()) == [1, 0]
    assert list(features.columns) == [column for column in EXPECTED_COLUMNS if column != TARGET_COLUMN]


def test_load_and_validate_benchmark_dataset_reads_csv(tmp_path: Path) -> None:
    source = tmp_path / "cs-training.csv"
    source.write_text(
        ",SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents\n"
        "1,1,0.7,45,2,0.8,9120,13,0,6,0,2\n",
        encoding="utf-8",
    )

    frame = load_and_validate_benchmark_dataset(source)

    assert TARGET_COLUMN in frame.columns
    assert "row_id" in frame.columns
    assert frame.loc[0, "row_id"] == 1