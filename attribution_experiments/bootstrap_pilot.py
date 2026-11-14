"""Collect one historical training rollout and create reviewable anchor preannotations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

from .pilot_inputs import MODELS, read_json
from .common import position_evidence as position


def parse_annotation(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("Annotation response contains no JSON object; inspect the retained raw response")
    annotation, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(annotation, dict) or annotation.get("classification") not in ("confirmed_agent_failure", "uncertain"):
        raise ValueError("Annotation must declare confirmed_agent_failure or uncertain")
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--annotate-only", action="store_true")
    args = parser.parse_args()
    selection = read_json(args.selection)
    if args.task_id not in selection["train_candidates"]:
        parser.error("Bootstrap can only access predeclared training candidates")
    if args.task_id in selection["dev"] or args.task_id in selection["pilot_eval"]:
        parser.error("Training candidates overlap held-out tasks")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TASK_MODEL_REQUEST_GATE_PATH", str(Path(os.environ.get("TMPDIR", root / ".codex-tmp")) / "request_gate.state"))
    sys.path[:0] = [str(root / "baselines/gepa/src"), str(root / "agent_runtime/mini-swe-agent-v2.4.6/src")]
    os.environ.update({"TASK_MODEL": MODELS["task"], "OPTIMIZER_MODEL": MODELS["optimizer"],
                       "ATTRIBUTION_REFERENCE_TRACE": str(args.reference_trace.resolve())})
    if os.environ.get("ALIYUN_API_KEY"):
        os.environ.setdefault("TASK_MODEL_API_KEY", os.environ["ALIYUN_API_KEY"])
        os.environ.setdefault("OPTIMIZER_API_KEY", os.environ["ALIYUN_API_KEY"])
    from .common.runner import run_task
    from .common.result_validity import retryable_result
    from .common.model_config import optimizer_api_key, optimizer_extra_body, OPTIMIZER_BASE_URL
    from gepa.lm import LM
    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet, filters=[("instance_id", "=", args.task_id)]).to_pylist()
    if len(rows) != 1:
        raise ValueError("Expected exactly one selected training task")
    directory = args.output.resolve() / "rollouts" / args.task_id
    if args.annotate_only:
        result = read_json(directory / "result.json")
    else:
        if directory.exists():
            raise FileExistsError(f"Existing rollout retained: {directory}. Use --annotate-only to label it")
        result = run_task(instance=rows[0], output_dir=directory,
                          udocker_repo=root / "native_baseline_runs/udocker", udocker_executable=Path("/unused"),
                          step_limit=250)
    if retryable_result(result) or result.get("exception") or not isinstance(result.get("evaluation", {}).get("resolved"), bool):
        raise RuntimeError("Bootstrap produced a non-gradeable result. Retain artifacts and repair infrastructure")
    if result["evaluation"]["resolved"]:
        print(json.dumps({"task_id": args.task_id, "resolved": True, "eligible_for_failure_training": False}))
        return
    trace_path = directory / "trajectory.json"
    trace = read_json(trace_path)
    steps = [{"step": number, "message_index": index} for number, index in enumerate(
        (index for index, message in enumerate(trace["messages"]) if message.get("role") == "assistant"), 1)]
    # Only observed training artifacts enter annotation. Hidden reference patches
    # and held-out task data are excluded from the prompt.
    transcript = [{key: message[key] for key in ("role", "content", "reasoning_content", "tool_calls", "tool_call_id") if key in message}
                  for message in trace["messages"]]
    prompt = """Create provisional failure-localization annotations for the supplied historical coding-agent failure.
Causing means an observed assistant decision that introduced or contributed to the failure.
Inevitable means the last observed recoverable decision opportunity under the same tools and remaining 250-call budget.
Do not assume that the last message is this opportunity. Use only the supplied observations.
If either anchor cannot be supported, return classification uncertain and explain the uncertainty.
Otherwise return a JSON object with classification confirmed_agent_failure and step_attribution containing
failure_causing_wrong_step and failure_inevitable_step. Each anchor must include step, message_index,
what_went_wrong, and evidence. step is one-based over assistant messages and message_index is zero-based
over all messages. The indices must identify the same assistant message. Add no repair instructions.
These labels will require independent review before being described as audited annotations.
Return only JSON.
""" + json.dumps({"task": rows[0]["problem_statement"], "assistant_positions": steps,
                   "messages": transcript, "evaluation": result["evaluation"],
                   "reference_harness": read_json(args.reference_trace)["info"]["config"]["agent"]}, ensure_ascii=False)
    model = LM("openai/" + MODELS["optimizer"], api_key=optimizer_api_key(), api_base=OPTIMIZER_BASE_URL,
               temperature=1.0, max_tokens=8192, reasoning_effort="none",
               extra_body=optimizer_extra_body() or {"enable_thinking": False})
    text = model(prompt).strip()
    response_dir = args.output.resolve() / "annotation_responses" / args.task_id
    response_dir.mkdir(parents=True, exist_ok=True)
    (response_dir / (uuid.uuid4().hex + ".json")).write_text(json.dumps({
        "model": MODELS["optimizer"], "response": text,
        "input_tokens": model.total_tokens_in, "output_tokens": model.total_tokens_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    annotation = parse_annotation(text)
    annotation.update({"schema_version": "2.0", "trace": {"path": str(trace_path)},
                       "review_status": "llm_preannotation_pending_independent_review",
                       "annotation_model": MODELS["optimizer"], "task_model": MODELS["task"],
                       "annotation_usage": {"input_tokens": model.total_tokens_in, "output_tokens": model.total_tokens_out}})
    if annotation["classification"] == "confirmed_agent_failure":
        for key in position.ANCHORS:
            anchor = annotation["step_attribution"][key]
            position.resolve_step(trace, {field: anchor[field] for field in ("step", "message_index")})
    output = args.output.resolve() / "analysis" / f"{args.task_id}.analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"task_id": args.task_id, "resolved": False, "classification": annotation["classification"],
                      "review_status": annotation["review_status"], "analysis": str(output)}))


if __name__ == "__main__":
    main()
