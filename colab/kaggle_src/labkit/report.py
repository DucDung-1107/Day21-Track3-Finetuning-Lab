"""Results I/O. Every run appends one row with the same columns, so the report table
writes itself and the grader can verify your numbers are internally consistent.

Kaggle deltas: the results directory is resolved from `$LAB_RESULTS` (falling back to
`./results`) instead of from this file's location. The repo copy computes
`parents[2]/"results"`, which is correct for `src/labkit/report.py` inside a checkout
and wrong for `/kaggle/working/labkit/report.py` — there it resolves to `/results`,
which is not writable, and the failure appears at the end of a 3-hour run. Plus
`hbar()`, so score comparisons are visible at a glance without a plotting dependency.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
from typing import Iterable


def results_dir_default() -> pathlib.Path:
    env = os.environ.get("LAB_RESULTS", "").strip()
    if env:
        return pathlib.Path(env)
    here = pathlib.Path(__file__).resolve()
    # In a repo checkout (`<root>/src/labkit/report.py` or
    # `<root>/colab/kaggle_src/labkit/report.py`) prefer the checkout's results dir.
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "requirements.txt").exists():
            return parent / "results"
    return pathlib.Path.cwd() / "results"


RESULTS = results_dir_default()


def append_row(row: dict, filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Append `row`, creating the header from its keys on first write.

    Rows with new keys are unioned into the header rather than silently dropped —
    losing a column because run 3 measured something run 1 did not is exactly the kind
    of quiet data loss that makes a results table untrustworthy.
    """
    out_dir = results_dir or results_dir_default()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))

    rows = existing + [row]
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def write_json(obj, filename: str, results_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = results_dir or results_dir_default()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_rows(filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> list[dict]:
    path = (results_dir or results_dir_default()) / filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def markdown_table(rows: Iterable[dict], columns: list[str] | None = None) -> str:
    """Paste-ready Markdown for REPORT.md."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_"
    cols = columns or list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def hbar(values: dict[str, float], width: int = 34, vmax: float | None = None,
         fmt: str = "{:.3f}") -> str:
    """A text bar chart, because a three-way score comparison read as a column of
    floats is where students stop noticing that regression collapsed.

    Text and not matplotlib on purpose: it survives being pasted into REPORT.md, a
    terminal, a PR comment or a grader's diff, and it adds no dependency to a notebook
    whose install cell is already the second-longest step in the lab.
    """
    if not values:
        return "_(no data)_"
    top = vmax if vmax is not None else max(max(values.values()), 1e-9)
    pad = max(len(k) for k in values)
    lines = []
    for k, v in values.items():
        filled = 0 if top <= 0 else max(0, min(width, round(width * v / top)))
        lines.append(f"{k:<{pad}}  {'█' * filled}{'·' * (width - filled)}  "
                     + fmt.format(v))
    return "\n".join(lines)

