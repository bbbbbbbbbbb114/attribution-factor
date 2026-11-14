import json
import sys

import pytest

from attribution_experiments import rq1_pilot
from attribution_experiments.pilot_inputs import MODELS
from attribution_experiments.gepa import run_native


@pytest.mark.parametrize("overrides, expected", [([], (3, 3)),
    (["--evolution-workers", "2", "--final-workers", "4"], (2, 4))])
def test_launcher_forwards_configured_workers(monkeypatch, capsys, overrides, expected):
    manifest = {"models": MODELS, "budget": {"seeds": [42], "step_limit": 250,
                "max_candidate_proposals": 4, "reflection_batch_size": 2},
                "paths": {"reference_trace": "/unused/reference.json", "analysis_root": "/unused/analysis"},
                "task_model_extra_body": {}, "optimizer_extra_body": {}, "splits": {"train": ["a"], "dev": ["b"], "pilot_eval": ["c"]}}
    monkeypatch.setattr(rq1_pilot, "load_bundle", lambda *args: (manifest, {}, {}))
    monkeypatch.setattr(rq1_pilot, "load_instances", lambda *args: {})
    monkeypatch.setenv("GEPA_EVOLUTION_WORKERS", "3")
    monkeypatch.setenv("GEPA_FINAL_WORKERS", "3")
    monkeypatch.setenv("EXPERIMENT_CONTAINER_BACKEND", "local_docker")
    for key in ("TASK_MODEL_BASE_URL", "OPTIMIZER_BASE_URL", "TASK_MODEL_API_KEY", "OPTIMIZER_API_KEY"):
        monkeypatch.setenv(key, "offline-value")
    received = []
    monkeypatch.setattr(run_native, "main", lambda: received.extend(sys.argv))
    monkeypatch.setattr(sys, "argv", ["pilot", "--bundle", "/unused/bundle", "--run", *overrides])
    rq1_pilot.main()
    report = json.loads(capsys.readouterr().out)
    assert report["parallelism"] == {"evolution_workers": expected[0], "final_workers": expected[1]}
    assert received[received.index("--evolution-workers") + 1] == str(expected[0])
    assert received[received.index("--final-workers") + 1] == str(expected[1])


def test_launcher_rejects_zero_workers_before_loading_inputs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pilot", "--bundle", "/unused", "--final-workers", "0"])
    with pytest.raises(SystemExit) as exc:
        rq1_pilot.main()
    assert exc.value.code == 2
