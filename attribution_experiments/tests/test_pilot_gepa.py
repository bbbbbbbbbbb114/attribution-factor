"""Exercise the native GEPA entrypoint with offline task/model responses."""

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from attribution_experiments import pilot_inputs as pilot
from attribution_experiments.gepa import run_native as native
from attribution_experiments.common import runner


@pytest.fixture
def pilot_run(tmp_path, monkeypatch):
    config = {"agent": {"system_template": "Base scaffold", "instance_template": "{{task}}"},
              "model": {"format_error_template": "{{error}}", "observation_template": "{{output.returncode}} {{output.output}}", "multimodal_regex": ""},
              "environment": {}}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"info": {"config": config}}))
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"messages": [{"role": "assistant", "content": "HISTORICAL_ONLY"},
                                              {"role": "tool", "content": "failure observed"}]}))
    analyses = tmp_path / "analysis"
    analyses.mkdir()
    (analyses / "train.analysis.json").write_text(json.dumps({
        "schema_version": "2.0", "classification": "confirmed_agent_failure", "trace": {"path": str(trace)},
        "step_attribution": {key: {"step": 1, "message_index": 0} for key in pilot.position.ANCHORS}}))
    parquet = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"instance_id": iid, "repo": "example/repo", "problem_statement": iid + "_PROBLEM",
                                         "patch": "HIDDEN_REFERENCE_PATCH", "test_patch": "HIDDEN_TEST_PATCH"}
                                        for iid in ("train", "dev", "heldout")]), parquet)
    cfg = tmp_path / "pilot.json"
    cfg.write_text(json.dumps({"models": pilot.MODELS, "parquet": str(parquet), "reference_trace": str(reference),
                               "analysis_root": str(analyses), "splits": {"train": ["train"], "dev": ["dev"], "pilot_eval": ["heldout"]},
                               "budget": {"max_candidate_proposals": 2, "reflection_batch_size": 1, "step_limit": 5, "seeds": [42]}}))
    bundle = tmp_path / "bundle"
    pilot.prepare(cfg, bundle)
    monkeypatch.setattr(native, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(native, "TASK_MODEL", pilot.MODELS["task"])
    monkeypatch.setattr(native, "OPTIMIZER_MODEL", pilot.MODELS["optimizer"])
    monkeypatch.setattr(runner, "REFERENCE_TRACE", reference)
    monkeypatch.setattr(native, "load_original_config", lambda: config)
    monkeypatch.setattr(native, "task_api_key", lambda: "offline-task-key")
    monkeypatch.setattr(native, "optimizer_api_key", lambda: "offline-optimizer-key")
    monkeypatch.setenv("EXPERIMENT_CONTAINER_BACKEND", "local_docker")
    monkeypatch.setenv("TASK_MODEL_EXTRA_BODY", "{}")
    monkeypatch.setenv("OPTIMIZER_EXTRA_BODY", "{}")
    calls, prompts = [], []

    def fake_task(*, instance, output_dir, **kwargs):
        calls.append((instance["instance_id"], output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trajectory.json").write_text(json.dumps({"messages": [{"role": "assistant", "content": "CURRENT_ONLY"}]}))
        (output_dir / "patch.diff").write_text("test patch")
        return {"exit_status": "Submitted", "evaluation": {"resolved": False}}

    class FakeLM:
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt):
            prompts.append(str(prompt))
            return f"```yaml\nupdates:\n  agent:\n    system_template: 'Proposed scaffold {len(prompts)}'\n  model: {{}}\n```"

    monkeypatch.setattr(native, "run_task", fake_task)
    monkeypatch.setattr(native, "LM", FakeLM)
    return bundle, calls, prompts, tmp_path / "runs"


@pytest.mark.parametrize("group", pilot.GROUPS)
def test_native_search_uses_disjoint_inputs_and_retains_h0_when_candidates_do_not_improve(pilot_run, monkeypatch, group):
    bundle, calls, prompts, runs = pilot_run
    monkeypatch.setattr(sys, "argv", ["gepa", "--ablation-group", group, "--run-name", "pilot",
                                     "--pilot-bundle", str(bundle), "--reflection-batch-size", "1",
                                     "--max-candidate-proposals", "2", "--step-limit", "5"])
    native.main()
    summary = json.loads((runs / group / "pilot/evolution_summary.json").read_text())
    assert summary["train_ids"] == ["train"] and summary["val_ids"] == ["dev"] and summary["final_ids"] == ["heldout"]
    assert summary["candidate_proposals_attempted"] == 2
    assert all(iid != "heldout" for iid, path in calls if "final" not in path.parts)
    assert len(prompts) == 2 and all("HISTORICAL_ONLY" in prompt for prompt in prompts)
    assert all("heldout_PROBLEM" not in prompt for prompt in prompts)
    assert all("HIDDEN_REFERENCE_PATCH" not in prompt and "HIDDEN_TEST_PATCH" not in prompt for prompt in prompts)
    assert all("Original Historical Training Trace" in prompt for prompt in prompts)
    assert all(("Provided Historical Position Evidence" in prompt) == (group != "baseline") for prompt in prompts)
    assert native.parse_candidate((runs / group / "pilot/best_candidate_overlay.yaml").read_text()) == {"agent": {}, "model": {}}


def test_initial_only_evaluates_without_optimizer_calls(pilot_run, monkeypatch):
    bundle, calls, prompts, runs = pilot_run
    monkeypatch.setattr(sys, "argv", ["gepa", "--ablation-group", "baseline", "--run-name", "h0",
                                     "--pilot-bundle", str(bundle), "--initial-only", "--reflection-batch-size", "1",
                                     "--max-candidate-proposals", "2", "--step-limit", "5"])
    monkeypatch.setattr(native, "optimizer_api_key", lambda: pytest.fail("H0 requested optimizer credentials"))
    native.main()
    assert not prompts and [iid for iid, _ in calls] == ["heldout", "train"]
    assert not (runs / "baseline/h0/engine").exists()
    assert json.loads((runs / "baseline/h0/h0_manifest.json").read_text())["optimizer_calls"] == 0
