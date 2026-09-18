"""Writer for the rebuilt tables: one CSV and one JSON per table under the results directory."""
from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lwl.paths import results_root

Row = dict[str, Any]


def table_paths(name: str) -> tuple[Path, Path]:
    """The CSV and JSON paths of one table."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"invalid table name: {name!r}")
    root = results_root()
    return root / f"{name}.csv", root / f"{name}.json"


def _cell(value: Any) -> Any:
    return "" if value is None else value


def write_table(name: str, rows: Sequence[Row], columns: Sequence[str]) -> tuple[Path, Path]:
    """Write results/<name>.csv and results/<name>.json; fields outside `columns` are dropped."""
    columns = list(columns)
    if not columns:
        raise ValueError(f"{name}: no columns to write")
    projected = [{column: row.get(column) for column in columns} for row in rows]
    csv_path, json_path = table_paths(name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in projected:
            writer.writerow({column: _cell(row[column]) for column in columns})
    payload = {"table": name, "columns": columns, "rows": projected}
    json_path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return csv_path, json_path
