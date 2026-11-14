"""Classify experiment artifacts that must be retried instead of graded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RETRYABLE_MARKERS = (
    "404 not found",
    "notfounderror",
    "remote disk below admission threshold",
    "remote transport failed",
    "remote execution completion unconfirmed",
    "this container run was invalidated by an infrastructure failure",
    "ssh connection failed",
    "connect to host 10.149.120.241",
    "timed out waiting for a resident slot",
    "remote execution failed; this task must be retried",
    "remote_infrastructure_error",
    "remoteinfrastructureerror",
    "workspace synchronization failed",
    "container stopped or disappeared",
    "overdue-payment",
    "allocated quota exceeded",
    "throttling.allocationquota",
)


def retryable_text(value: Any) -> bool:
    """Return whether a status/exception describes infrastructure failure."""
    if value is None:
        return False
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    lowered = value.lower()
    return any(marker in lowered for marker in RETRYABLE_MARKERS)


def retryable_result(result: dict[str, Any]) -> bool:
    """Return whether a task result is invalid and must not count as complete."""
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("resolved") is None:
        return True
    fields = {
        "exit_status": result.get("exit_status"),
        "exception": result.get("exception"),
        "traceback": result.get("traceback"),
        "error": result.get("error"),
        "status": result.get("status"),
        "evaluation": evaluation,
    }
    return retryable_text(fields)


def recreate_final_problems(output_dir: Path, results: list[dict]) -> dict[str, str]:
    """Require a real matching evaluator report, including for negative grades."""
    problems = {}
    for row in results:
        iid = row['instance_id']
        try:
            report = json.loads((output_dir / iid / 'evaluation/eval_result.json').read_text())
        except (OSError, ValueError):
            problems[iid] = 'missing_or_unreadable_evaluation'
            continue
        if (report.get('instance_id') != iid or not isinstance(report.get('resolved'), bool)
                or report['resolved'] != row.get('resolved')):
            problems[iid] = 'evaluation_summary_mismatch'
        elif retryable_result({'evaluation': report}) or retryable_result(row):
            problems[iid] = 'retryable_infrastructure_result'
    return problems
