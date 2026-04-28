"""Project-wide paths and runtime settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_RANDOM_STATE = 42
PROJECT_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
    LOGS_DIR,
]


def ensure_project_directories() -> list[Path]:
    """Create the standard project folders if they do not exist."""

    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
    return PROJECT_DIRECTORIES
