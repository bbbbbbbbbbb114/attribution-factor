"""Run native GEPA evolution over the full mutable mini-swe text scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from jinja2 import Environment, TemplateSyntaxError

from gepa.lm import LM
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from attribution_experiments.common.analysis_payloads import (
    DEFAULT_PAYLOAD_ROOT,
    GROUPS,
    load_payloads,
    model_visible_payload,
    snapshot_payloads,
)

from attribution_experiments.common.dataset import (
    load_failed_instances,
    select_repository_balanced_instances,
)
from attribution_experiments.common.model_config import (
    OPTIMIZER_BASE_URL,
    OPTIMIZER_LITELLM_MODEL,
    OPTIMIZER_MODEL,
    TASK_MODEL,
    TASK_STEP_LIMIT,
    optimizer_api_key,
    task_api_key,
)
from attribution_experiments.common.runner import load_original_config, run_task
from attribution_experiments.common.result_validity import retryable_result
from attribution_experiments.gepa.metric_resume import MetricReplay
from attribution_experiments.gepa.protocol import proposal_budget, require_completed_evolution


WORKSPACE = Path(__file__).resolve().parents[2]
RUN_ROOT = WORKSPACE / "attribution_experiments" / "gepa"
UDOCKER_REPO = WORKSPACE / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = WORKSPACE / "native_baseline_runs" / ".venv" / "bin" / "udocker"
AGENT_TEXT_FIELDS = ("system_template", "instance_template")
MODEL_TEXT_FIELDS = ("format_error_template", "observation_template", "multimodal_regex")


def build_seed_candidate() -> str:
    return yaml.safe_dump(
        {"updates": {"agent": {}, "model": {}}},
        sort_keys=False,
        width=10000,
    )


def build_reflection_prompt_template(*, position_only: bool = False) -> str:
    original = load_original_config()
    base_scaffold = {
        "agent": {key: original["agent"][key] for key in AGENT_TEXT_FIELDS},
        "model": {key: original["model"][key] for key in MODEL_TEXT_FIELDS},
    }
    rendered_base = yaml.safe_dump(base_scaffold, sort_keys=False, width=10000)
    prompt = f"""You are optimizing the complete mutable text scaffold of mini-swe-agent.

Optimization goal: improve SWE-bench solve rate for a fixed task model
model using the existing bash tool and submission protocol.

IMMUTABLE BASE SCAFFOLD START
{rendered_base}
IMMUTABLE BASE SCAFFOLD END

CURRENT SPARSE YAML OVERLAY START
<curr_param>
CURRENT SPARSE YAML OVERLAY END

EVALUATION FEEDBACK START
<side_info>
EVALUATION FEEDBACK END

A "Provided Failure Analysis" section in the feedback is trusted pre-computed
failure localization. It names the exact agent step that caused the failure or
made it unrecoverable. Prefer it over re-reading the raw trajectory when deciding
which scaffold field to change, and derive a general prompt rule from that
localization rather than a task-specific fix.

Propose one general improvement based on the feedback. All five fields are
searchable, but include only fields you actually change. The output schema is:

updates:
  agent: {{}}
  model: {{}}

Allowed agent keys: system_template, instance_template.
Allowed model keys: format_error_template, observation_template, multimodal_regex.

Hard interface requirements:
- A replacement instance_template MUST contain the exact Jinja expression with
  two opening and two closing braces around task, exactly as in the base.
- A replacement format_error_template MUST contain the exact Jinja expression
  with two opening and two closing braces around error.
- A replacement observation_template MUST preserve output.returncode and
  output.output as Jinja expressions and must remain valid Jinja.
- Preserve the bash tool and exact COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  submission contract.
- Do not mention task IDs, repository names, issue facts, expected patches, or
  hidden answers.
- Do not copy unchanged fields.

Return ONLY the YAML overlay inside one fenced code block. No explanation outside
the block. The top-level mapping has exactly two sections, agent and model. Use
{{}} for an unchanged section, for example "agent: {{}}". Do not append extra
closing braces after the top-level mapping."""
    if position_only:
        from attribution_experiments.common.position_evidence import INSTRUCTIONS
        start = prompt.index('A "Provided Failure Analysis" section')
        end = prompt.index('Propose one general improvement', start)
        prompt = prompt[:start] + INSTRUCTIONS + '\n\n' + prompt[end:]
    return prompt


def parse_candidate(candidate: str) -> dict[str, dict[str, str]]:
    text = candidate.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        artifact = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        stripped = text.rstrip()
        if stripped.endswith("}"):
            try:
                artifact = yaml.safe_load(stripped[:-1].rstrip())
            except yaml.YAMLError:
                raise ValueError(f"Candidate is not valid YAML: {exc}") from exc
        else:
            raise ValueError(f"Candidate is not valid YAML: {exc}") from exc
    if isinstance(artifact, dict) and set(artifact) <= {"agent", "model"} and "updates" not in artifact:
        artifact = {"updates": artifact}
    elif isinstance(artifact, dict) and "updates" in artifact and set(artifact) != {"updates"}:
        updates = artifact.get("updates")
        if isinstance(updates, dict):
            for key, value in artifact.items():
                if key != "updates" and key in {"agent", "model"} and isinstance(value, dict):
                    updates.setdefault(key, value)
            artifact = {"updates": updates}
    if not isinstance(artifact, dict) or set(artifact) != {"updates"}:
        raise ValueError("Candidate must contain exactly the top-level key: updates")
    raw_updates = artifact["updates"]
    if not isinstance(raw_updates, dict):
        raise ValueError("Candidate updates must be a mapping")
    unknown_sections = set(raw_updates) - {"agent", "model"}
    if unknown_sections:
        raise ValueError(f"Unknown candidate sections: {sorted(unknown_sections)}")
    updates = {
        "agent": raw_updates.get("agent", {}),
        "model": raw_updates.get("model", {}),
    }

    allowed = {"agent": set(AGENT_TEXT_FIELDS), "model": set(MODEL_TEXT_FIELDS)}
    for section, values in updates.items():
        if not isinstance(values, dict):
            raise ValueError(f"Candidate updates.{section} must be a mapping")
        unknown = set(values) - allowed[section]
        if unknown:
            raise ValueError(f"Unknown updates.{section} fields: {sorted(unknown)}")
        for key, template in values.items():
            if not isinstance(template, str):
                raise ValueError(f"Candidate updates.{section}.{key} must be a string")
            if key != "multimodal_regex" and not template.strip():
                raise ValueError(f"Candidate updates.{section}.{key} must not be empty")
            try:
                Environment().parse(template)
            except TemplateSyntaxError as exc:
                raise ValueError(
                    f"Invalid Jinja template in updates.{section}.{key}: {exc}"
                ) from exc
    return updates


def resolve_candidate(candidate: str) -> dict[str, dict[str, str]]:
    updates = parse_candidate(candidate)
    original = load_original_config()
    resolved = {
        "agent": {key: original["agent"][key] for key in AGENT_TEXT_FIELDS},
        "model": {key: original["model"][key] for key in MODEL_TEXT_FIELDS},
    }
    for section, values in updates.items():
        resolved[section].update(values)
    compact_instance = resolved["agent"]["instance_template"].replace(" ", "")
    if "{{task}}" not in compact_instance:
        raise ValueError("Resolved agent.instance_template must contain exact {{task}}")
    compact_error = resolved["model"]["format_error_template"].replace(" ", "")
    if "{{error}}" not in compact_error:
        raise ValueError("Resolved model.format_error_template must contain exact {{error}}")
    compact_observation = resolved["model"]["observation_template"].replace(" ", "")
    for required in ("output.returncode", "output.output"):
        if required not in compact_observation:
            raise ValueError(
                f"Resolved model.observation_template must preserve {required}"
            )
    return resolved


def compact_agent_trace(messages: list[dict], max_events: int = 18) -> list[dict]:
    events: list[dict] = []
    for index, message in enumerate(messages):
        role = message.get("role", "")
        event: dict = {"step": index, "role": role}
        if role == "assistant":
            reasoning = message.get("reasoning_content") or message.get("content") or ""
            if reasoning.strip():
                event["reasoning"] = reasoning.strip()[:500]
            actions = message.get("extra", {}).get("actions", [])
            if actions:
                event["commands"] = [
                    str(action.get("command", ""))[:500] for action in actions
                ]
        elif role in {"tool", "user"}:
            content = str(message.get("content", ""))
            if content:
                event["observation"] = (
                    content[:500] + ("..." + content[-200:] if len(content) > 700 else "")
                )
        elif role == "exit":
            event["exit_status"] = message.get("extra", {}).get("exit_status", "")
            event["content"] = str(message.get("content", ""))[:300]
        else:
            continue
        events.append(event)

    if len(events) <= max_events:
        return events
    keep_front = max_events // 2
    keep_back = max_events - keep_front
    return [
        *events[:keep_front],
        {"role": "elided", "count": len(events) - max_events},
        *events[-keep_back:],
    ]



class NativeEvaluator:
    def __init__(
        self,
        *,
        output_dir: Path,
        api_key: str,
        step_limit: int,
        analysis_payloads: dict[str, dict] | None = None,
        metric_replay: MetricReplay | None = None,
        historical_traces: dict[str, dict] | None = None,
    ):
        self.output_dir = output_dir
        self.api_key = api_key
        self.step_limit = step_limit
        self.analysis_payloads = analysis_payloads or {}
        self.metric_replay = metric_replay
        self.historical_traces = historical_traces or {}
        self._counter = 0
        self._lock = threading.Lock()
        self.evaluated_candidates: set[str] = set()

    def _next_dir(self, candidate: str, instance_id: str) -> Path:
        with self._lock:
            call_id = self._counter
            self._counter += 1
        candidate_id = hashlib.sha256(candidate.encode()).hexdigest()[:12]
        return self.output_dir / "metric_calls" / f"{call_id:04d}_{candidate_id}_{instance_id}"

    def __call__(self, candidate: str, instance: dict) -> tuple[float, dict]:
        call_dir = self._next_dir(candidate, instance["instance_id"])
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "candidate_scaffold.yaml").write_text(candidate)
        try:
            artifact = resolve_candidate(candidate)
            (call_dir / "resolved_scaffold.yaml").write_text(
                yaml.safe_dump(artifact, sort_keys=False, width=10000)
            )
        except ValueError as exc:
            error = str(exc)
            (call_dir / "candidate_error.json").write_text(
                json.dumps({"instance_id": instance["instance_id"], "error": error}, indent=2)
            )
            return 0.0, {
                "Inputs": {
                    "Task ID": instance["instance_id"],
                    "Problem": instance["problem_statement"][:200],
                },
                "Generated Outputs": {"Candidate": candidate[:2000]},
                "Feedback": {"Status": "invalid candidate", "Validation Error": error},
                "scores": {"correctness": 0.0},
            }

        run = self.metric_replay.restore(candidate, instance["instance_id"], call_dir) if self.metric_replay else None
        attempts = max(1, int(os.environ.get("GEPA_INFRA_ATTEMPTS", "2")))
        for attempt in range(attempts):
            if run is None:
                attempt_dir = call_dir if attempt == 0 else call_dir / f"infrastructure_retry_{attempt}"
                run = run_task(
                    instance=instance,
                    output_dir=attempt_dir,
                    udocker_repo=UDOCKER_REPO,
                    udocker_executable=UDOCKER_EXECUTABLE,
                    api_key=self.api_key,
                    agent_overrides=artifact["agent"],
                    model_overrides=artifact["model"],
                    step_limit=self.step_limit,
                )
                if attempt and not retryable_result(run):
                    shutil.copytree(attempt_dir, call_dir, dirs_exist_ok=True)
            if not retryable_result(run) or attempt + 1 == attempts:
                break
            print(
                f"Retrying GEPA infrastructure result for {instance['instance_id']} "
                f"({attempt + 1}/{attempts - 1})",
                flush=True,
            )
            run = None
        if retryable_result(run):
            raise RuntimeError(
                f"GEPA metric call produced a non-gradeable infrastructure result for "
                f"{instance['instance_id']}: {run.get('exit_status')}: {run.get('exception', '')}"
            )
        evaluation = run.get("evaluation", {})
        score = 1.0 if evaluation.get("resolved") else 0.0
        trajectory = json.loads((call_dir / "trajectory.json").read_text())
        patch = (call_dir / "patch.diff").read_text()
        test_output_path = call_dir / "evaluation" / "test_output.txt"
        test_output = test_output_path.read_text() if test_output_path.exists() else evaluation.get("error", "")
        compact_test_output = (
            f"{test_output[:500]}...{test_output[-500:]}" if len(test_output) > 1000 else test_output
        )
        info = {}
        if instance["instance_id"] in self.historical_traces:
            info["Original Historical Training Trace"] = self.historical_traces[instance["instance_id"]]
        payload = self.analysis_payloads.get(instance["instance_id"])
        visible = model_visible_payload(payload) if payload is not None else {}
        if visible:
            label = "Provided Historical Position Evidence" if self.historical_traces else "Provided Failure Analysis (Trusted Evidence)"
            info[label] = visible
            from attribution_experiments.common.flow_evidence import has_flow, packet
            if has_flow(payload):
                info["Provided Failure Analysis (Trusted Evidence)"] = {
                    "historical_original_failure": packet(payload, instance['instance_id']),
                    "current_candidate_rollout": {
                        "exit_status": run.get('exit_status'), "resolved": bool(score),
                        "trace_location": "Generated Outputs / Agent Trace",
                        "instruction": "Compare current actions/observations and the candidate scaffold with the historical control/data dependencies. Historical step numbers do not identify steps in this new rollout. Cite supporting or contradicted graph relations in reflection, and propose only general scaffold changes. A current success can mean the historical mechanism no longer applies."
                    },
                }
        info["Inputs"] = {
            "Task ID": instance["instance_id"],
            "Problem": instance["problem_statement"][:200],
        }
        info["Generated Outputs"] = {
            "Patch": patch[:1000],
            "Agent Trace": compact_agent_trace(trajectory["messages"]),
        }
        info["Feedback"] = {
            "Status": "resolved" if score else evaluation.get("error", "unresolved"),
            "Test Output": compact_test_output,
        }
        info["scores"] = {"correctness": score}
        with self._lock:
            self.evaluated_candidates.add(candidate)
        return score, info


def extract_best_candidate(result) -> str:
    candidate = result.best_candidate
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        if len(candidate) != 1:
            raise ValueError(f"Expected one GEPA artifact component, got {list(candidate)}")
        return next(iter(candidate.values()))
    raise TypeError(f"Unsupported GEPA candidate type: {type(candidate)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--evolution-workers", type=int, default=1)
    parser.add_argument("--final-workers", type=int, default=1)
    parser.add_argument("--adaptation-tasks", type=int, default=32)
    parser.add_argument("--train-tasks", type=int, default=10)
    parser.add_argument('--full-trace-pass', action='store_true',
                        help='Cover all 148 training tasks in one native minibatch epoch; selection set overlaps training')
    parser.add_argument("--reflection-batch-size", type=int, default=3)
    parser.add_argument("--validation-offset", type=int)
    parser.add_argument("--validation-tasks", type=int)
    parser.add_argument("--adaptation-seed", type=int, default=20260902)
    parser.add_argument("--max-candidate-proposals", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-metrics-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-bundle", type=Path,
                        help="Frozen RQ1 train/dev/pilot_eval inputs")
    parser.add_argument("--initial-only", action="store_true",
                        help="Independently evaluate H0 on pilot_eval without optimization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-limit", type=int, default=TASK_STEP_LIMIT)
    args = parser.parse_args()
    if args.evolution_workers < 1:
        parser.error("--evolution-workers must be at least 1")
    if args.final_workers < 1:
        parser.error("--final-workers must be at least 1")
    if not args.pilot_bundle and args.adaptation_tasks < 2:
        parser.error("--adaptation-tasks must be at least 2")
    if not args.pilot_bundle and not args.full_trace_pass and not 1 <= args.train_tasks < args.adaptation_tasks:
        parser.error('--train-tasks must be positive and smaller than --adaptation-tasks')
    if not args.pilot_bundle and not 1 <= args.reflection_batch_size <= (args.adaptation_tasks if args.full_trace_pass else args.train_tasks):
        parser.error('--reflection-batch-size must be between 1 and --train-tasks')
    if args.max_candidate_proposals < 2:
        parser.error("--max-candidate-proposals must be at least 2")
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run-name must be a single directory name")
    if args.step_limit < 1:
        parser.error("--step-limit must be positive")
    if args.initial_only and (not args.pilot_bundle or args.ablation_group != "baseline"):
        parser.error("--initial-only requires --pilot-bundle and the baseline group")
    if args.initial_only and args.resume:
        parser.error("Use a fresh H0 run directory for an independent rerun")
    if args.pilot_bundle and (args.smoke or args.full_trace_pass or args.resume_metrics_from):
        parser.error("Pilot bundles require fresh disjoint data and independent metrics")
    optimizer_key = optimizer_api_key() if not args.initial_only else None
    task_key = task_api_key()

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise RuntimeError("GEPA evolution requires a fresh run directory; preserve old results and use a new --run-name")
    from attribution_experiments.common.execution_protocol import record_execution_backend
    historical_traces = {}
    pilot = None
    if args.pilot_bundle:
        from attribution_experiments.pilot_inputs import load_bundle, load_instances
        from attribution_experiments.common.runner import REFERENCE_TRACE
        pilot, analysis_payloads, historical_traces = load_bundle(args.pilot_bundle, args.ablation_group)
        if pilot['models'] != {'task': TASK_MODEL, 'optimizer': OPTIMIZER_MODEL}:
            parser.error('Model environment differs from frozen pilot models')
        if Path(pilot['paths']['reference_trace']).resolve() != REFERENCE_TRACE.resolve():
            parser.error('Set ATTRIBUTION_REFERENCE_TRACE to the frozen pilot reference trace before launch')
        actual_budget = {'max_candidate_proposals': args.max_candidate_proposals,
                         'reflection_batch_size': args.reflection_batch_size, 'step_limit': args.step_limit}
        if any(value != pilot['budget'][key] for key, value in actual_budget.items()) or args.seed not in pilot['budget']['seeds']:
            parser.error('Launch arguments differ from the frozen pilot budget')
        if json.loads(os.environ.get('TASK_MODEL_EXTRA_BODY', '{}')) != pilot['task_model_extra_body']:
            parser.error('Task-model request parameters differ from the frozen pilot inputs')
        split_instances = load_instances(pilot)
        trainset, valset, final_instances = (split_instances[key] for key in ('train', 'dev', 'pilot_eval'))
        adaptation_instances = trainset + valset
        instances = adaptation_instances + final_instances
        step_limit = args.step_limit
        if args.reflection_batch_size > len(trainset):
            parser.error('--reflection-batch-size exceeds pilot training count')
        if args.initial_only:
            run_dir.mkdir(parents=True, exist_ok=True)
            record_execution_backend(run_dir)
            (run_dir / 'h0_manifest.json').write_text(json.dumps({
                'models': pilot['models'], 'sources': pilot['source_sha256'],
                'splits': pilot['splits'], 'step_limit': step_limit, 'seed': args.seed,
                'optimizer_calls': 0}, indent=2))
            evaluate_final(build_seed_candidate(), final_instances, run_dir, task_key, step_limit, args.final_workers)
            evaluate_final(build_seed_candidate(), trainset, run_dir / 'historical_recovery', task_key, step_limit, args.final_workers)
            return
    else:
        analysis_payloads = load_payloads(args.ablation_group, args.payload_root.resolve())
    from attribution_experiments.common.flow_evidence import write_receipt
    if not args.pilot_bundle:
        instances = load_failed_instances()
    if args.full_trace_pass and (args.smoke or args.adaptation_tasks != len(instances)):
        parser.error('--full-trace-pass requires all 148 tasks and a non-smoke run')
    if args.pilot_bundle:
        pass
    elif args.smoke:
        adaptation_instances = instances[:2]
        trainset, valset = adaptation_instances[:1], adaptation_instances[1:]
        step_limit = 3
        final_instances = instances[:1]
    else:
        if args.adaptation_tasks > len(instances):
            parser.error(f"--adaptation-tasks cannot exceed {len(instances)}")
        adaptation_instances = select_repository_balanced_instances(
            instances,
            limit=args.adaptation_tasks,
            seed=args.adaptation_seed,
        )
        from .protocol import split_adaptation
        if args.full_trace_pass:
            trainset = adaptation_instances
            offset = 10 if args.validation_offset is None else args.validation_offset
            size = 22 if args.validation_tasks is None else args.validation_tasks
            if offset < 0 or size < 1 or offset + size > len(instances):
                parser.error('Invalid candidate-selection subset')
            selection_pool = select_repository_balanced_instances(instances, limit=offset + size,
                                                                  seed=args.adaptation_seed)
            valset = selection_pool[offset:]
            from .protocol import one_pass_proposals
            expected = one_pass_proposals(len(trainset), args.reflection_batch_size)
            if args.max_candidate_proposals != expected:
                parser.error(f'One full trace pass requires exactly {expected} proposals at this minibatch size')
        else:
            trainset, valset = split_adaptation(adaptation_instances, args.train_tasks,
                                              validation_offset=args.validation_offset,
                                              validation_tasks=args.validation_tasks)
        step_limit = args.step_limit
        final_instances = instances

    reflection_batch_size = 1 if args.smoke else args.reflection_batch_size
    from attribution_experiments.common.optimization_budget import pin_budget
    budget = {
        'protocol': 'gepa_checkpoint_budget_v3', 'group': args.ablation_group,
        'train_ids': [x['instance_id'] for x in trainset],
        'val_ids': [x['instance_id'] for x in valset],
        'max_candidate_proposals': args.max_candidate_proposals,
        'reflection_batch_size': reflection_batch_size, 'step_limit': step_limit,
        'full_trace_pass': args.full_trace_pass,
        'batch_sampler': 'single_pass_unpadded_v1' if args.full_trace_pass else 'epoch_shuffled',
        'analysis_sha256': hashlib.sha256(json.dumps(analysis_payloads, sort_keys=True).encode()).hexdigest(),
    }
    if pilot:
        budget.update({'pilot_inputs_sha256': pilot['inputs_sha256'], 'pilot_eval_ids': pilot['splits']['pilot_eval'],
                       'source_sha256': pilot['source_sha256'], 'models': pilot['models'], 'seed': args.seed,
                       'task_model_extra_body': pilot['task_model_extra_body']})
    pin_budget(run_dir, budget, has_progress=(run_dir / 'engine').exists() or any((run_dir / 'metric_calls').glob('*/result.json')))
    record_execution_backend(run_dir)
    snapshot_payloads(run_dir, args.ablation_group, analysis_payloads)
    write_receipt(run_dir, analysis_payloads, 'native reflection side-info; historical/current evidence separated')
    max_metric_calls = proposal_budget(
        len(valset),
        reflection_batch_size,
        args.max_candidate_proposals,
    )
    batch_sampler = 'epoch_shuffled'
    if args.full_trace_pass:
        from .single_pass_sampler import SinglePassBatchSampler
        from .protocol import one_pass_metric_budget
        batch_sampler = SinglePassBatchSampler(reflection_batch_size, seed=args.seed)
        max_metric_calls = one_pass_metric_budget(len(valset), len(trainset), reflection_batch_size)
    metric_replay = MetricReplay(
        args.resume_metrics_from, run_dir, model=TASK_MODEL, step_limit=step_limit
    ) if args.resume_metrics_from else None
    if metric_replay:
        print(f"Validated GEPA metrics available for replay: {metric_replay.available}", flush=True)
    evaluator = NativeEvaluator(
        output_dir=run_dir,
        api_key=task_key,
        step_limit=step_limit,
        analysis_payloads=analysis_payloads,
        metric_replay=metric_replay,
        historical_traces=historical_traces,
    )
    if args.resume:
        from .protocol import prepare_engine_resume
        evaluator._counter = prepare_engine_resume(run_dir)
        for path in (run_dir / 'metric_calls').glob('*/result.json'):
            if isinstance(json.loads(path.read_text()).get('evaluation', {}).get('resolved'), bool):
                evaluator.evaluated_candidates.add((path.parent / 'candidate_scaffold.yaml').read_text())
    reflection_lm = LM(
        OPTIMIZER_LITELLM_MODEL,
        api_key=optimizer_key,
        api_base=OPTIMIZER_BASE_URL,
        temperature=1.0,
        max_tokens=8192,
        reasoning_effort="none",
    )
    seed_artifact = build_seed_candidate()
    result = optimize_anything(
        seed_candidate=seed_artifact,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_dir / "engine"),
                max_metric_calls=max_metric_calls,
                max_candidate_proposals=args.max_candidate_proposals,
                max_workers=args.evolution_workers,
                parallel=True,
                acceptance_criterion="strict_improvement",
                write_agent_state=True,
                seed=args.seed,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=reflection_batch_size,
                batch_sampler=batch_sampler,
                reflection_prompt_template=build_reflection_prompt_template(position_only=bool(pilot)),
                skip_perfect_score=False,
            ),
            merge=None,
            refiner=None,
        ),
    )

    proposal_attempts = len([
        path for path in (run_dir / "engine" / "iterations").glob("*/meta.json")
        if path.parent.name != "seed"
    ])
    from .protocol import reflection_coverage
    coverage = reflection_coverage(run_dir / 'engine', [x['instance_id'] for x in trainset])
    (run_dir / 'reflection_coverage.json').write_text(json.dumps(coverage, indent=2))
    if args.full_trace_pass and (coverage['missing_task_ids'] or coverage['repeated_task_ids']):
        raise RuntimeError('Full trace pass must cover every training task exactly once: '
                           + json.dumps(coverage))
    evolution_check = {
        "optimization_completed": True,
        "resume_policy": "preserve_upstream_engine_checkpoint",
        "reflection_batch_size": reflection_batch_size,
        "reflection_coverage": coverage,
        "full_trace_pass": args.full_trace_pass,
        "batch_sampler": 'single_pass_unpadded_v1' if args.full_trace_pass else 'epoch_shuffled',
        "selection_overlaps_training": bool({x['instance_id'] for x in trainset} & {x['instance_id'] for x in valset}),
        "candidate_proposals_attempted": proposal_attempts,
        "metric_artifact_count_including_restarts": len(list((run_dir / 'metric_calls').glob('*/result.json'))),
        "new_candidates_evaluated": len({
            json.dumps(resolve_candidate(candidate), sort_keys=True)
            for candidate in evaluator.evaluated_candidates
        } - {json.dumps(resolve_candidate(seed_artifact), sort_keys=True)}),
    }
    if not pilot:
        require_completed_evolution(evolution_check)
    if proposal_attempts < args.max_candidate_proposals:
        raise RuntimeError(
            f"GEPA stopped after {proposal_attempts} proposals; expected "
            f"{args.max_candidate_proposals}. Do not finalize a truncated search."
        )
    best_artifact = extract_best_candidate(result)
    resolved_best = resolve_candidate(best_artifact)
    (run_dir / "best_candidate_overlay.yaml").write_text(best_artifact)
    (run_dir / "best_scaffold.yaml").write_text(
        yaml.safe_dump(resolved_best, sort_keys=False, width=10000)
    )
    summary = {
        **evolution_check,
        "budget_protocol": "gepa_checkpoint_budget_v3",
        "max_candidate_proposals": args.max_candidate_proposals,
        "best_idx": result.best_idx,
        "best_score": result.best_score,
        "total_evals": result.total_metric_calls,
        "metric_calls_this_process": result.total_evals,
        "candidate_count": len(result.candidates),
        "max_metric_calls": max_metric_calls,
        "evolution_workers": args.evolution_workers,
        "step_limit": step_limit,
        "ablation_group": args.ablation_group,
        "task_model": TASK_MODEL,
        "optimizer_model": OPTIMIZER_MODEL,
        "analysis_json_injected": args.ablation_group != "baseline",
        "analysis_payloads_injected_at": (
            "none" if args.ablation_group == "baseline" else "GEPA reflection side_info as trusted evidence"
        ),
        "candidate_encoding": "sparse YAML overlay resolved against the original scaffold",
        "adaptation_protocol": "repository_balanced_explicit_split_v2",
        "adaptation_tasks": len(adaptation_instances),
        "adaptation_task_ids": [x["instance_id"] for x in adaptation_instances],
        "adaptation_seed": args.adaptation_seed,
        "candidate_surface": {
            "agent": list(AGENT_TEXT_FIELDS),
            "model": list(MODEL_TEXT_FIELDS),
            "protected": ["model identity", "environment", "resource cap", "tasks", "evaluator"],
        },
        "train_ids": [x["instance_id"] for x in trainset],
        "val_ids": [x["instance_id"] for x in valset],
        "final_ids": [x["instance_id"] for x in final_instances],
        "seed": args.seed,
        "pilot_protocol": pilot['protocol'] if pilot else None,
        "optimizer_usage_this_process": {"input_tokens": reflection_lm.total_tokens_in,
                            "output_tokens": reflection_lm.total_tokens_out,
                            "reported_cost": reflection_lm.total_cost},
    }
    (run_dir / "evolution_summary.json").write_text(json.dumps(summary, indent=2))

    evaluate_final(best_artifact, final_instances, run_dir, task_key, step_limit, args.final_workers)
    if pilot:
        evaluate_final(best_artifact, trainset, run_dir / 'historical_recovery', task_key, step_limit, args.final_workers)


def evaluate_final(artifact, instances, run_dir, task_key, step_limit, workers):
    final_evaluator = NativeEvaluator(output_dir=run_dir / "final", api_key=task_key, step_limit=step_limit)

    def final_rollout(instance: dict) -> dict:
        score, info = final_evaluator(artifact, instance)
        return {
            "instance_id": instance["instance_id"],
            "resolved": bool(score),
            "status": info["Feedback"]["Status"],
        }

    final_results_by_id = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(final_rollout, instance): instance["instance_id"]
            for instance in instances
        }
        for future in as_completed(futures):
            task_id = futures[future]
            final_results_by_id[task_id] = future.result()
    final_results = [
        final_results_by_id[item["instance_id"]] for item in instances
    ]
    final_summary = {
        "resolved": sum(item["resolved"] for item in final_results),
        "total": len(final_results),
        "workers": workers,
        "results": final_results,
    }
    (run_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2))
    print(json.dumps({"final": final_summary}, indent=2))


if __name__ == "__main__":
    main()
