"""Text renderings of result tables."""
from __future__ import annotations

import io

import pandas as pd

FORMATS = ("text", "markdown", "csv")


def _cell(value) -> str:
    if isinstance(value, float):
        return "nan" if pd.isna(value) else f"{value:.2f}"
    return str(value)


def _rows(table: pd.DataFrame):
    header = [table.index.name or "", *table.columns]
    body = [[str(label), *(_cell(value) for value in row)] for label, row in zip(table.index, table.itertuples(index=False))]
    return header, body


def render(table: pd.DataFrame, fmt: str = "text") -> str:
    """Render a table whose index holds the row labels."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    header, body = _rows(table)
    if fmt == "csv":
        buffer = io.StringIO()
        for row in (header, *body):
            buffer.write(",".join(row) + "\n")
        return buffer.getvalue().rstrip("\n")
    widths = [max(len(row[i]) for row in (header, *body)) for i in range(len(header))]
    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)

    def line(row):
        first = row[0].ljust(widths[0])
        rest = [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        return "  ".join([first, *rest]).rstrip()

    return "\n".join(line(row) for row in (header, *body))
