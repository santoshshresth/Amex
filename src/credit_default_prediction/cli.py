"""Command line entry points for project setup and checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_DIRECTORIES, PROJECT_ROOT, ensure_project_directories
from .data_loading.benchmark import load_and_validate_benchmark_dataset, split_features_target


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="credit-default",
        description="Bootstrap and validate the credit default prediction project layout.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("init", "check", "benchmark"),
        default="check",
        help="Use 'init' to create folders or 'check' to verify the layout.",
    )
    return parser


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_init() -> int:
    """Create the standard repository folders."""

    created = ensure_project_directories()
    print("Project directories are ready:")
    for directory in created:
        print(f" - {_relative(directory)}")
    return 0


def run_check() -> int:
    """Report whether the expected repository layout is present."""

    missing = [directory for directory in PROJECT_DIRECTORIES if not directory.exists()]
    if missing:
        print("Project layout is incomplete:")
        for directory in missing:
            print(f" - missing {_relative(directory)}")
        print("Run 'credit-default init' or 'python -m credit_default_prediction init'.")
        return 1

    print("Project layout is ready.")
    for directory in PROJECT_DIRECTORIES:
        print(f" - {_relative(directory)}")
    return 0


def run_benchmark() -> int:
    """Load and summarize the benchmark dataset."""

    frame = load_and_validate_benchmark_dataset()
    features, target = split_features_target(frame)
    feature_columns = [column for column in features.columns if column != "row_id"]

    target_rate = float(target.mean()) if len(target) else 0.0
    missing_cells = int(frame.isna().sum().sum())

    print("Benchmark dataset is ready.")
    print(f" - rows: {len(frame)}")
    print(f" - features: {len(feature_columns)}")
    print(f" - target default rate: {target_rate:.4f}")
    print(f" - missing cells: {missing_cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the package script and module execution."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return run_init()
    if args.command == "benchmark":
        return run_benchmark()
    return run_check()
