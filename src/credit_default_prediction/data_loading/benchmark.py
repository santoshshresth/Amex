"""Load and validate the GiveMeSomeCredit benchmark dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import RAW_DATA_DIR

TARGET_COLUMN = "SeriousDlqin2yrs"
ROW_ID_COLUMN = "row_id"
EXPECTED_COLUMNS = [
    TARGET_COLUMN,
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
DEFAULT_BENCHMARK_FILENAMES = ("cs-training.csv", "cs-training (1).csv")


def resolve_benchmark_path(path: str | Path | None = None) -> Path:
    """Resolve the benchmark file path, checking the project raw-data folder first."""

    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Benchmark file not found: {candidate}")

    for filename in DEFAULT_BENCHMARK_FILENAMES:
        candidate = RAW_DATA_DIR / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(str(RAW_DATA_DIR / filename) for filename in DEFAULT_BENCHMARK_FILENAMES)
    raise FileNotFoundError(f"Could not find the benchmark dataset. Looked for: {searched}")


def load_benchmark_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the benchmark dataset into a dataframe with normalized column names."""

    dataset_path = resolve_benchmark_path(path)
    frame = pd.read_csv(dataset_path, na_values=["", "NA", "NaN", "nan"])

    unnamed_columns = [column for column in frame.columns if str(column).startswith("Unnamed")]
    if unnamed_columns:
        first_unnamed = unnamed_columns[0]
        frame = frame.rename(columns={first_unnamed: ROW_ID_COLUMN})

    return frame


def validate_benchmark_dataset(frame: pd.DataFrame) -> None:
    """Validate the benchmark dataset schema and target values."""

    errors: list[str] = []

    if frame.empty:
        errors.append("dataset is empty")

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"missing expected columns: {', '.join(missing_columns)}")

    if TARGET_COLUMN in frame.columns:
        target_values = set(frame[TARGET_COLUMN].dropna().unique().tolist())
        invalid_values = sorted(value for value in target_values if value not in {0, 1})
        if invalid_values:
            errors.append(f"target column contains invalid values: {invalid_values}")

    if errors:
        raise ValueError("Invalid benchmark dataset: " + "; ".join(errors))


def load_and_validate_benchmark_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the benchmark dataset and raise if the schema is not valid."""

    frame = load_benchmark_dataset(path)
    validate_benchmark_dataset(frame)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the benchmark dataset into features and target."""

    validate_benchmark_dataset(frame)
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype("Int64")
    return features, target


def describe_columns(columns: Iterable[str]) -> str:
    """Format a column list for error messages."""

    return ", ".join(columns)