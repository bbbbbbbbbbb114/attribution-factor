"""Check or launch one frozen GEPA pilot condition in a fresh process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .pilot_inputs import GROUPS, load_bundle, load_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute after preflight passes")
    args = parser.parse_args()
    if args.initial_only and args.group != "baseline":
        parser.error("H0 uses the baseline group")
    manifest, _, _ = load_bundle(args.bundle.resolve(), args.group)
    budget = manifest["budget"]
    if args.seed not in budget["seeds"]:
        parser.error("Select a seed listed in the frozen budget")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TASK_MODEL_REQUEST_GATE_PATH", str(Path(os.environ.get("TMPDIR", root / ".codex-tmp")) / "request_gate.state"))
    sys.path[:0] = [str(root / "baselines/gepa/src"),
                    str(root / "agent_runtime/mini-swe-agent-v2.4.6/src")]
    os.environ.update({"TASK_MODEL": manifest["models"]["task"],
                       "OPTIMIZER_MODEL": manifest["models"]["optimizer"],
                       "ATTRIBUTION_REFERENCE_TRACE": manifest["paths"]["reference_trace"],
                       "ATTRIBUTION_ANALYSIS_ROOT": manifest["paths"]["analysis_root"],
                       "ATTRIBUTION_TASK_IDS": str(args.bundle.resolve() / "train_ids.txt"),
                       "TASK_MODEL_EXTRA_BODY": json.dumps(manifest["task_model_extra_body"]),
                       "OPTIMIZER_EXTRA_BODY": json.dumps(manifest["optimizer_extra_body"])})
    # A shared provider key can be supplied through the launching process.
    shared_key = os.environ.get("ALIYUN_API_KEY")
    if shared_key:
        os.environ.setdefault("TASK_MODEL_API_KEY", shared_key)
        os.environ.setdefault("OPTIMIZER_API_KEY", shared_key)
    blockers = []
    if sys.platform != "linux":
        blockers.append("The existing fcntl/container runtime requires a Linux execution host")
    settings = ["TASK_MODEL_BASE_URL", "TASK_MODEL_API_KEY"]
    if not args.initial_only:
        settings.extend(["OPTIMIZER_BASE_URL", "OPTIMIZER_API_KEY"])
    for name in settings:
        if not os.environ.get(name):
            blockers.append(f"Missing environment setting: {name}")
    backend = os.environ.get("EXPERIMENT_CONTAINER_BACKEND")
    if backend not in ("local_docker", "remote_docker"):
        blockers.append("Set EXPERIMENT_CONTAINER_BACKEND=local_docker or remote_docker")
    if backend == "remote_docker" and not os.environ.get("EXPERIMENT_REMOTE_HOST"):
        blockers.append("Set EXPERIMENT_REMOTE_HOST to the Docker execution host")
    for package in ("yaml", "jinja2", "pydantic", "litellm", "pyarrow", "swebench", "cloudpickle", "tqdm"):
        if importlib.util.find_spec(package) is None:
            blockers.append(f"Missing Python dependency: {package}")
    storage = Path(os.environ.get("UDOCKER_RUNTIME_ROOT", root / "attribution_experiments/runtime/containers")).resolve()
    if sys.platform == "linux" and not storage.is_relative_to("/volume/med"):
        blockers.append("Existing runtime requires UDOCKER_RUNTIME_ROOT under /volume/med")
    if not blockers:
        load_instances(manifest)
    report = {"ready": not blockers, "blockers": blockers, "models": manifest["models"],
              "split_sizes": {key: len(value) for key, value in manifest["splits"].items()},
              "budget": budget, "group": args.group, "seed": args.seed,
              "initial_only": args.initial_only, "checks": "local inputs and dependencies"}
    print(json.dumps(report, indent=2), flush=True)
    if blockers:
        raise SystemExit(2)
    if not args.run:
        return
    tag = "h0" if args.initial_only else args.group
    name = f"rq1_{args.bundle.resolve().name}_{tag}_seed{args.seed}"
    from .gepa import run_native
    sys.argv = [str(Path(run_native.__file__)), "--ablation-group", args.group,
                "--run-name", name, "--pilot-bundle", str(args.bundle.resolve()),
                "--seed", str(args.seed), "--step-limit", str(budget["step_limit"]),
                "--max-candidate-proposals", str(budget["max_candidate_proposals"]),
                "--reflection-batch-size", str(budget["reflection_batch_size"])]
    if args.initial_only:
        sys.argv.append("--initial-only")
    run_native.main()


if __name__ == "__main__":
    main()
