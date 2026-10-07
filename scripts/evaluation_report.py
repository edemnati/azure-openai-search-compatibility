"""Generate privacy-conscious HTML reports for current-versus-target evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

METRICS = (
    ("groundedness", "Groundedness"),
    ("relevance", "Relevance"),
    ("coherence", "Coherence"),
    ("fluency", "Fluency"),
    ("similarity", "Similarity"),
    ("f1_score", "F1"),
)


def write_html_report(
    *,
    output_path: Path,
    timestamp: str,
    current_result: Mapping[str, Any],
    target_result: Mapping[str, Any],
    comparison: Mapping[str, Mapping[str, float]],
) -> None:
    """Write aggregate and per-case scores without prompts, answers, or contexts."""
    current_rows = _rows_by_id(current_result)
    target_rows = _rows_by_id(target_result)
    case_ids = list(current_rows)
    if set(case_ids) != set(target_rows):
        raise ValueError("Current and target evaluation results contain different case IDs")

    aggregate_rows = "".join(
        _aggregate_row(metric, label, comparison) for metric, label in METRICS
    )
    case_rows = "".join(
        _case_row(case_id, current_rows[case_id], target_rows[case_id])
        for case_id in case_ids
    )
    wins, ties, regressions = _outcome_counts(comparison)
    metric_headers = "".join(
        f'<th colspan="3">{escape(label)}</th>' for _, label in METRICS
    )
    metric_subheaders = "".join(
        "<th>Current</th><th>Target</th><th>Delta</th>" for _ in METRICS
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Azure OpenAI Search compatibility evaluation</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; }}
    body {{ margin: 0; background: #f5f7fa; color: #172033; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 32px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .subtitle, .notice {{ color: #536174; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr));
              gap: 16px; margin: 24px 0; }}
    .card {{ background: white; border: 1px solid #dfe5ec; border-radius: 10px;
             padding: 18px; box-shadow: 0 2px 8px #1720330d; }}
    .card strong {{ display: block; font-size: 28px; margin-top: 6px; }}
    .table-wrap {{ overflow-x: auto; background: white; border: 1px solid #dfe5ec;
                   border-radius: 10px; margin: 16px 0 28px; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{ border-bottom: 1px solid #e7ebf0; padding: 10px 12px; text-align: right; }}
    th {{ background: #eef3f8; color: #33445c; }}
    th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0;
                                     background: inherit; }}
    tbody tr:nth-child(even) {{ background: #fafbfd; }}
    .positive {{ color: #08783e; font-weight: 600; }}
    .negative {{ color: #b42318; font-weight: 600; }}
    .neutral {{ color: #536174; }}
    .pass {{ color: #08783e; }}
    .fail {{ color: #b42318; }}
    footer {{ margin-top: 28px; color: #68778c; font-size: 13px; }}
    @media (max-width: 760px) {{
      main {{ padding: 18px; }}
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Current OYD vs target compatibility bridge</h1>
  <p class="subtitle">Evaluation run {escape(timestamp)} UTC</p>
  <p class="notice">This report intentionally excludes questions, answers, ground truth,
    retrieved context, intents, endpoints, and credentials. Case IDs and evaluator scores
    are included.</p>

  <section class="cards" aria-label="Evaluation summary">
    <div class="card">Golden cases<strong>{len(case_ids)}</strong></div>
    <div class="card">Target improvements<strong class="positive">{wins}</strong></div>
    <div class="card">Ties<strong class="neutral">{ties}</strong></div>
    <div class="card">Regressions<strong class="negative">{regressions}</strong></div>
  </section>

  <h2>Aggregate metrics</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Metric</th><th>Current</th><th>Target</th><th>Delta</th>
        <th>Current pass</th><th>Target pass</th></tr></thead>
      <tbody>{aggregate_rows}</tbody>
    </table>
  </div>

  <h2>Per-case scores</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th rowspan="2">Case ID</th>{metric_headers}</tr>
        <tr>{metric_subheaders}</tr>
      </thead>
      <tbody>{case_rows}</tbody>
    </table>
  </div>

  <footer>Generated locally by the Azure OpenAI Search compatibility evaluation runner.</footer>
</main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def _rows_by_id(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = result.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Evaluation result does not contain a rows list")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Evaluation result contains an invalid row")
        case_id = row.get("inputs.id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Evaluation row does not contain a case ID")
        if case_id in rows_by_id:
            raise ValueError(f"Duplicate evaluation case ID: {case_id}")
        rows_by_id[case_id] = row
    return rows_by_id


def _aggregate_row(
    metric: str,
    label: str,
    comparison: Mapping[str, Mapping[str, float]],
) -> str:
    score = comparison.get(f"{metric}.{metric}_score", {})
    passed = comparison.get(f"{metric}.{metric}_passed", {})
    return (
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{_number(score.get('current'))}</td>"
        f"<td>{_number(score.get('target'))}</td>"
        f"{_delta_cell(score.get('delta'))}"
        f"<td>{_percentage(passed.get('current'))}</td>"
        f"<td>{_percentage(passed.get('target'))}</td>"
        "</tr>"
    )


def _case_row(
    case_id: str,
    current: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str:
    cells = [f"<td>{escape(case_id)}</td>"]
    for metric, _ in METRICS:
        key = f"outputs.{metric}.{metric}_score"
        current_score = _numeric(current.get(key))
        target_score = _numeric(target.get(key))
        delta = (
            target_score - current_score
            if current_score is not None and target_score is not None
            else None
        )
        cells.extend(
            (
                f"<td>{_number(current_score)}</td>",
                f"<td>{_number(target_score)}</td>",
                _delta_cell(delta),
            )
        )
    return f"<tr>{''.join(cells)}</tr>"


def _outcome_counts(
    comparison: Mapping[str, Mapping[str, float]],
) -> tuple[int, int, int]:
    deltas = [
        comparison.get(f"{metric}.{metric}_score", {}).get("delta")
        for metric, _ in METRICS
    ]
    numeric_deltas = [value for value in deltas if isinstance(value, (int, float))]
    wins = sum(value > 0 for value in numeric_deltas)
    ties = sum(value == 0 for value in numeric_deltas)
    regressions = sum(value < 0 for value in numeric_deltas)
    return wins, ties, regressions


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _number(value: Any) -> str:
    numeric = _numeric(value)
    return "&mdash;" if numeric is None else f"{numeric:.3f}"


def _percentage(value: Any) -> str:
    numeric = _numeric(value)
    return "&mdash;" if numeric is None else f"{numeric:.0%}"


def _delta_cell(value: Any) -> str:
    numeric = _numeric(value)
    if numeric is None:
        return "<td>&mdash;</td>"
    style = "positive" if numeric > 0 else "negative" if numeric < 0 else "neutral"
    return f'<td class="{style}">{numeric:+.3f}</td>'
