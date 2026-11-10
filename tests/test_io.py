import pandas as pd
import pytest

from pvpipe.io import DataError, read_dataset, read_frame, write_dataset, write_frame


def write(tmp_path, text, name="production.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip_keeps_values_and_timestamps(tmp_path, hours):
    frame = pd.DataFrame({"power_kw": [0.0, 1.23456, 99.9]}, index=hours(periods=3))
    write_frame(frame, tmp_path / "p.csv")
    back = read_frame(tmp_path / "p.csv", ["power_kw"])
    assert list(back.index) == list(frame.index)
    assert list(back["power_kw"]) == [0.0, 1.235, 99.9]


def test_other_offsets_are_converted_to_utc(tmp_path):
    path = write(tmp_path, "timestamp,power_kw\n2024-06-01T02:00:00+02:00,5\n")
    frame = read_frame(path, ["power_kw"])
    assert frame.index[0] == pd.Timestamp("2024-06-01T00:00:00Z")


def test_rows_are_sorted_by_time(tmp_path):
    path = write(tmp_path, "timestamp,power_kw\n2024-06-01T01:00:00Z,2\n2024-06-01T00:00:00Z,1\n")
    assert list(read_frame(path, ["power_kw"])["power_kw"]) == [1, 2]


def test_timestamp_without_offset_names_the_line(tmp_path):
    path = write(tmp_path, "timestamp,power_kw\n2024-06-01T00:00:00Z,1\n2024-06-01T01:00:00,2\n")
    with pytest.raises(DataError, match="line 3: timestamp has no UTC offset"):
        read_frame(path, ["power_kw"])


def test_missing_column(tmp_path):
    path = write(tmp_path, "timestamp,kw\n2024-06-01T00:00:00Z,1\n")
    with pytest.raises(DataError, match="missing columns: power_kw"):
        read_frame(path, ["power_kw"])


def test_duplicate_timestamp(tmp_path):
    path = write(tmp_path, "timestamp,power_kw\n2024-06-01T00:00:00Z,1\n2024-06-01T00:00:00Z,2\n")
    with pytest.raises(DataError, match="duplicate timestamp"):
        read_frame(path, ["power_kw"])


def test_text_in_a_value_column_becomes_missing(tmp_path):
    path = write(tmp_path, "timestamp,power_kw\n2024-06-01T00:00:00Z,n/a\n2024-06-01T01:00:00Z,4\n")
    values = read_frame(path, ["power_kw"])["power_kw"]
    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == 4


def test_dataset_round_trip(tmp_path, dataset):
    write_dataset(dataset, tmp_path / "d")
    back = read_dataset(tmp_path / "d")
    assert len(back.production) == len(dataset.production)
    assert list(back.weather_forecast.columns) == ["ghi", "temp_air", "cloud_cover"]
    assert back.weather_observed is not None


def test_observed_weather_is_optional(tmp_path, dataset):
    write_dataset(dataset, tmp_path / "d")
    (tmp_path / "d" / "weather_observed.csv").unlink()
    assert read_dataset(tmp_path / "d").weather_observed is None


def test_dataset_directory_must_exist(tmp_path):
    with pytest.raises(DataError, match="not a directory"):
        read_dataset(tmp_path / "nowhere")
