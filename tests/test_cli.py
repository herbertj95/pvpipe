import pandas as pd
import pytest

from pvpipe.cli import main
from tests.conftest import DEMO_SITE

SITE = ["--site", str(DEMO_SITE)]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data") / "demo"
    assert main(["simulate", *SITE, "--data", str(directory), "--start", "2024-03-01", "--days", "60", "--seed", "4"]) == 0
    return directory


@pytest.fixture(scope="module")
def dirty_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data") / "dirty"
    main(["simulate", *SITE, "--data", str(directory), "--start", "2024-03-01", "--days", "60", "--seed", "4", "--dirty"])
    return directory


def test_simulate_writes_three_files(data_dir, capsys):
    assert sorted(path.name for path in data_dir.iterdir()) == [
        "production.csv",
        "weather_forecast.csv",
        "weather_observed.csv",
    ]
    production = pd.read_csv(data_dir / "production.csv")
    assert len(production) == 60 * 24
    assert production["timestamp"].iloc[0] == "2024-03-01T00:00:00+00:00"


def test_qc_on_clean_data(data_dir, capsys):
    assert main(["qc", *SITE, "--data", str(data_dir)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "0 of 1440 hours flagged (0.0 %)"


def test_qc_on_dirty_data(dirty_dir, capsys):
    assert main(["qc", *SITE, "--data", str(dirty_dir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["flag", "hours"]
    assert dict(line.split() for line in lines[1:6]) == {
        "missing": "48",
        "negative": "0",
        "above_limit": "1",
        "night_production": "1",
        "stuck": "5",
    }
    assert lines[-1] == "55 of 1440 hours flagged (3.8 %)"


def test_backtest_prints_one_row_per_model(data_dir, capsys):
    assert main(["backtest", *SITE, "--data", str(data_dir), "--format", "csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "model,hours,mae_kw,rmse_kw,mbe_kw,nmae_pct,nrmse_pct,skill"
    assert [line.split(",")[0] for line in lines[1:]] == [
        "persistence",
        "smart_persistence",
        "climatology",
        "physical",
        "ridge",
    ]
    scores = {line.split(",")[0]: float(line.split(",")[-1]) for line in lines[1:]}
    assert scores["persistence"] == 0.0
    assert scores["ridge"] > 0.2


def test_backtest_model_selection_and_reference(data_dir, capsys):
    args = ["backtest", *SITE, "--data", str(data_dir), "--models", "physical,ridge", "--reference", "physical", "--format", "csv"]
    assert main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(",")[0] for line in lines[1:]] == ["physical", "ridge"]
    assert lines[1].endswith(",0.00")


def test_backtest_survives_dirty_data(dirty_dir, capsys):
    assert main(["backtest", *SITE, "--data", str(dirty_dir), "--models", "persistence,ridge"]) == 0
    assert "ridge" in capsys.readouterr().out


def test_backtest_writes_the_forecasts(data_dir, tmp_path, capsys):
    out = tmp_path / "out" / "forecasts.csv"
    assert main(["backtest", *SITE, "--data", str(data_dir), "--models", "persistence", "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written.columns) == ["timestamp", "persistence", "actual"]
    assert len(written) == 30 * 24


def test_backtest_reference_must_be_selected(data_dir, capsys):
    assert main(["backtest", *SITE, "--data", str(data_dir), "--models", "ridge"]) == 1
    assert "reference model 'persistence' must be one of --models" in capsys.readouterr().err


def test_backtest_with_too_little_history(data_dir, capsys):
    assert main(["backtest", *SITE, "--data", str(data_dir), "--train-days", "60"]) == 1
    assert "need more than 60 days" in capsys.readouterr().err


def test_forecast_prints_24_hours(data_dir, capsys):
    assert main(["forecast", *SITE, "--data", str(data_dir), "--date", "2024-04-20"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "timestamp,power_kw"
    assert len(lines) == 25
    assert lines[1] == "2024-04-20T00:00:00+00:00,0.0"
    assert max(float(line.split(",")[1]) for line in lines[1:]) > 50


def test_forecast_for_a_day_outside_the_data(data_dir, capsys):
    assert main(["forecast", *SITE, "--data", str(data_dir), "--date", "2025-01-01"]) == 1
    assert "no weather forecast for 2025-01-01" in capsys.readouterr().err


def test_unknown_model_is_a_usage_error(data_dir, capsys):
    with pytest.raises(SystemExit) as raised:
        main(["backtest", *SITE, "--data", str(data_dir), "--models", "persistence,arima"])
    assert raised.value.code == 2
    assert "unknown model(s): arima" in capsys.readouterr().err


def test_missing_site_file(tmp_path, capsys):
    assert main(["qc", "--site", str(tmp_path / "none.toml"), "--data", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_dataset(tmp_path, capsys):
    assert main(["qc", *SITE, "--data", str(tmp_path / "none")]) == 1
    assert "not a directory" in capsys.readouterr().err
