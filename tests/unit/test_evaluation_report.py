"""Tests for the privacy-conscious evaluation HTML report."""

from pathlib import Path

from scripts.evaluation_report import write_html_report


def test_report_contains_scores_without_customer_content(tmp_path: Path) -> None:
    current = {
        "rows": [
            {
                "inputs.id": "case-<one>",
                "inputs.query": "PRIVATE QUESTION",
                "inputs.response": "PRIVATE CURRENT ANSWER",
                "inputs.context": "PRIVATE CONTEXT",
                "outputs.groundedness.groundedness_score": 4.0,
                "outputs.relevance.relevance_score": 3.0,
            }
        ]
    }
    target = {
        "rows": [
            {
                "inputs.id": "case-<one>",
                "inputs.query": "PRIVATE QUESTION",
                "inputs.response": "PRIVATE TARGET ANSWER",
                "inputs.context": "PRIVATE CONTEXT",
                "outputs.groundedness.groundedness_score": 5.0,
                "outputs.relevance.relevance_score": 3.0,
            }
        ]
    }
    comparison = {
        "groundedness.groundedness_score": {
            "current": 4.0,
            "target": 5.0,
            "delta": 1.0,
        },
        "groundedness.groundedness_passed": {
            "current": 0.5,
            "target": 1.0,
            "delta": 0.5,
        },
        "relevance.relevance_score": {
            "current": 3.0,
            "target": 3.0,
            "delta": 0.0,
        },
    }
    output_path = tmp_path / "report.html"

    write_html_report(
        output_path=output_path,
        timestamp="20261006-160500",
        current_result=current,
        target_result=target,
        comparison=comparison,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "case-&lt;one&gt;" in report
    assert "PRIVATE QUESTION" not in report
    assert "PRIVATE CURRENT ANSWER" not in report
    assert "PRIVATE TARGET ANSWER" not in report
    assert "PRIVATE CONTEXT" not in report
    assert "Target improvements" in report
    assert "+1.000" in report
    assert "100%" in report
