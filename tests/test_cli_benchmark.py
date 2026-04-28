from __future__ import annotations

from pathlib import Path

from credit_default_prediction import cli
from credit_default_prediction import config


def test_run_benchmark_prints_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "cs-training.csv"
    source.write_text(
        ",SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents\n"
        "1,1,0.7,45,2,0.8,9120,13,0,6,0,2\n"
        "2,0,0.2,50,0,0.1,2600,4,0,0,0,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "load_and_validate_benchmark_dataset", lambda: __import__("credit_default_prediction.data_loading.benchmark", fromlist=["load_and_validate_benchmark_dataset"]).load_and_validate_benchmark_dataset(source))

    exit_code = cli.run_benchmark()

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Benchmark dataset is ready." in captured
    assert "rows: 2" in captured
    assert "features: 10" in captured