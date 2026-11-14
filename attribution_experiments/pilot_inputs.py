"""Freeze and verify the three RQ1 conditions without requiring flow graphs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from .common import position_evidence as position

GROUPS = ("baseline", "group1_1_causing", "group1_2_inevitable")
SPLITS = ("train", "dev", "pilot_eval")
MODELS = {"task": "qwen3.6-35b-a3b", "optimizer": "deepseek-v4.1-flash-1017"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_splits(splits: dict) -> None:
    if set(splits) != set(SPLITS):
        raise ValueError("Supply exactly train, dev and pilot_eval task lists")
    seen = set()
    for name in SPLITS:
        ids = splits[name]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name} must be a nonempty task list")
        for iid in ids:
            if not isinstance(iid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", iid):
                raise ValueError(f"Invalid task ID in {name}")
            if iid in seen:
                raise ValueError(f"Duplicate or overlapping task ID: {iid}")
            seen.add(iid)


def validate_budget(budget: dict, training_count: int) -> None:
    for key, minimum in (("max_candidate_proposals", 2), ("reflection_batch_size", 1), ("step_limit", 1)):
        if type(budget.get(key)) is not int or budget[key] < minimum:
            raise ValueError(f"Invalid pilot budget: {key}")
    if budget["reflection_batch_size"] > training_count:
        raise ValueError("Reflection batch exceeds training count")
    seeds = budget.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct integer optimization seeds")


def source_paths(config: dict, base: Path) -> dict[str, Path]:
    return {name: (base / config[name]).resolve() for name in
            ("parquet", "reference_trace", "analysis_root")}


def build_training_inputs(config: dict, paths: dict) -> tuple[dict, dict, dict]:
    payloads = {group: {} for group in GROUPS}
    historical = {}
    sources = {}
    stats = {"tasks": 0, "same_anchor": 0, "inevitable_at_last_step": 0,
             "anchor_distances": [], "excerpt_chars": {g: [] for g in GROUPS[1:]},
             "truncated": {g: 0 for g in GROUPS[1:]}, "review_status": {}}
    for iid in config["splits"]["train"]:
        analysis_path = paths["analysis_root"] / f"{iid}.analysis.json"
        analysis = read_json(analysis_path)
        if analysis.get("schema_version") != "2.0" or analysis.get("classification") != "confirmed_agent_failure":
            raise ValueError(f"Expected confirmed failure analysis schema 2.0 for {iid}")
        trace_path = Path(analysis["trace"]["path"])
        if not trace_path.is_absolute():
            # Match the existing analysis_payloads.trace_path convention.
            trace_path = Path(__file__).resolve().parents[1] / trace_path
        trace_path = trace_path.resolve()
        raw = trace_path.read_bytes()
        trace = json.loads(raw)
        sha = position.digest(raw)
        sources[str(analysis_path)] = position.digest(analysis_path.read_bytes())
        sources[str(trace_path)] = sha
        anchors = analysis["step_attribution"]
        selected = {key: {field: anchors[key][field] for field in ("step", "message_index")}
                    for key in position.ANCHORS}
        # All conditions receive the same original messages in reflection feedback.
        # Copy the same message fields used by position.resolve_step, excluding config.
        messages = []
        for index, message in enumerate(trace["messages"]):
            clean = {key: copy.deepcopy(message[key]) for key in
                     ("role", "content", "reasoning_content", "tool_calls", "tool_call_id") if key in message}
            actions = (message.get("extra") or {}).get("actions")
            if actions is not None:
                clean["actions"] = copy.deepcopy(actions)
            messages.append({"message_index": index, "message": clean})
        historical[iid] = {"sha256": sha, "messages": messages,
                           "interpretation": "Original historical trajectory. Indices do not identify current rollout steps."}
        for group, key in zip(GROUPS, (None, *position.ANCHORS)):
            info = {} if key is None else {key: selected[key]}
            payload = position.convert_payload({"task_id": iid, "provided_information": info}, trace, sha)
            position.verify_payload(payload, source_trace=trace)
            payloads[group][iid] = payload
            if key is not None:
                excerpt = payload["provided_information"]["historical_step_evidence"][key]
                stats["excerpt_chars"][group].append(len(excerpt["raw_excerpt"]))
                stats["truncated"][group] += int(excerpt["truncated"])
        causing, inevitable = (selected[key]["step"] for key in position.ANCHORS)
        stats["tasks"] += 1
        stats["same_anchor"] += int(causing == inevitable)
        stats["anchor_distances"].append(abs(causing - inevitable))
        stats["inevitable_at_last_step"] += int(inevitable == sum(m.get("role") == "assistant" for m in trace["messages"]))
        stats["review_status"][iid] = analysis.get("review_status", "unreported")
    return {"payloads": payloads, "historical_traces": historical}, sources, stats


def prepare(config_path: Path, output: Path) -> dict:
    config = read_json(config_path)
    validate_splits(config["splits"])
    validate_budget(config["budget"], len(config["splits"]["train"]))
    if config["models"] != MODELS:
        raise ValueError(f"This pilot uses {MODELS}")
    for key in ("task_model_extra_body", "optimizer_extra_body"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a JSON object")
    paths = source_paths(config, config_path.resolve().parent)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs: " + ", ".join(missing))
    reference = read_json(paths["reference_trace"])["info"]["config"]
    for section in ("agent", "model", "environment"):
        if not isinstance(reference[section], dict):
            raise ValueError(f"Invalid reference config section: {section}")
    inputs, sources, stats = build_training_inputs(config, paths)
    for name in ("parquet", "reference_trace"):
        sources[str(paths[name])] = position.digest(paths[name].read_bytes())
    inputs_raw = (json.dumps(inputs, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest = {"protocol": "rq1_pilot_v1", "evidence_protocol": position.PROTOCOL,
                "models": config["models"], "splits": config["splits"], "groups": list(GROUPS), "budget": config["budget"],
                "task_model_extra_body": config.get("task_model_extra_body", {}),
                "optimizer_extra_body": config.get("optimizer_extra_body", {}),
                "paths": {key: str(value) for key, value in paths.items()},
                "source_sha256": sources, "inputs_sha256": position.digest(inputs_raw),
                "anchor_statistics": stats}
    if output.exists():
        raise FileExistsError(f"Use a fresh bundle directory: {output}")
    output.mkdir(parents=True)
    (output / "inputs.json").write_bytes(inputs_raw)
    (output / "train_ids.txt").write_text("\n".join(config["splits"]["train"]) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(root: Path, group: str) -> tuple[dict, dict, dict]:
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"] != "rq1_pilot_v1" or group not in GROUPS:
        raise ValueError("Pilot protocol/group mismatch")
    validate_splits(manifest["splits"])
    validate_budget(manifest["budget"], len(manifest["splits"]["train"]))
    if manifest["models"] != MODELS or manifest["groups"] != list(GROUPS):
        raise ValueError("Pilot model/condition mismatch")
    if (root / "train_ids.txt").read_text(encoding="utf-8").splitlines() != manifest["splits"]["train"]:
        raise ValueError("Frozen training task list changed")
    raw = (root / "inputs.json").read_bytes()
    if position.digest(raw) != manifest["inputs_sha256"]:
        raise ValueError("Frozen pilot inputs changed")
    for name, sha in manifest["source_sha256"].items():
        if position.digest(Path(name).read_bytes()) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    inputs = json.loads(raw)
    train_ids = set(manifest["splits"]["train"])
    if set(inputs["historical_traces"]) != train_ids or set(inputs["payloads"][group]) != train_ids:
        raise ValueError("Pilot evidence must contain exactly the training tasks")
    for iid, payload in inputs["payloads"][group].items():
        position.verify_payload(payload)
        expected = {} if group == "baseline" else {position.ANCHORS[GROUPS.index(group) - 1]: None}
        actual = {key: None for key in position.ANCHORS if key in payload["provided_information"]}
        if actual != expected or payload["task_id"] != iid:
            raise ValueError("Pilot treatment contents do not match condition")
    return manifest, inputs["payloads"][group], inputs["historical_traces"]


def load_instances(manifest: dict) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    ids = [iid for split in SPLITS for iid in manifest["splits"][split]]
    rows = pq.read_table(manifest["paths"]["parquet"], filters=[("instance_id", "in", ids)]).to_pylist()
    by_id = {row["instance_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("Parquet must contain every selected task exactly once")
    return {name: [by_id[iid] for iid in manifest["splits"][name]] for name in SPLITS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.config, args.output)
    print(json.dumps({"bundle": str(args.output.resolve()), "tasks": {key: len(value) for key, value in manifest["splits"].items()},
                      "anchor_statistics": manifest["anchor_statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
