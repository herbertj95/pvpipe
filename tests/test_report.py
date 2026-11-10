import pandas as pd
import pytest

from pvpipe.report import render


@pytest.fixture
def table():
    frame = pd.DataFrame(
        {"hours": [10, 8], "mae_kw": [1.5, 12.345], "skill": [0.0, float("nan")]},
        index=pd.Index(["persistence", "ridge"], name="model"),
    )
    return frame


def test_text_is_aligned_with_two_decimals(table):
    assert render(table).splitlines() == [
        "model        hours  mae_kw  skill",
        "persistence     10    1.50   0.00",
        "ridge            8   12.35    nan",
    ]


def test_markdown(table):
    assert render(table, "markdown").splitlines() == [
        "| model | hours | mae_kw | skill |",
        "|---|---|---|---|",
        "| persistence | 10 | 1.50 | 0.00 |",
        "| ridge | 8 | 12.35 | nan |",
    ]


def test_csv(table):
    assert render(table, "csv") == "model,hours,mae_kw,skill\npersistence,10,1.50,0.00\nridge,8,12.35,nan"


def test_unknown_format(table):
    with pytest.raises(ValueError, match="format must be one of"):
        render(table, "html")
