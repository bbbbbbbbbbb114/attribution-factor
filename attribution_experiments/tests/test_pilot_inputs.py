import copy
import json

import pytest

from attribution_experiments import pilot_inputs as pilot


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def config_file(tmp_path):
    trace = {"info": {"api_key": "SECRET_CONFIG"}, "messages": [
        {"role": "user", "content": "problem"},
        {"role": "assistant", "content": "ORIGINAL_EDIT", "extra": {"actions": [{"command": "edit"}], "api_key": "SECRET_EXTRA"}},
        {"role": "tool", "content": "TEST_FAILED"},
        {"role": "assistant", "content": "ORIGINAL_SUBMIT"}]}
    trace_path = tmp_path / "trace.json"
    write_json(trace_path, trace)
    analysis = {"schema_version": "2.0", "classification": "confirmed_agent_failure",
                "trace": {"path": str(trace_path)}, "step_attribution": {
                    pilot.position.ANCHORS[0]: {"step": 1, "message_index": 1, "what_went_wrong": "ANNOTATED_DIAGNOSIS"},
                    pilot.position.ANCHORS[1]: {"step": 2, "message_index": 3}}}
    write_json(tmp_path / "analysis/train-1.analysis.json", analysis)
    write_json(tmp_path / "reference.json", {"info": {"config": {"agent": {}, "model": {}, "environment": {}}}})
    (tmp_path / "tasks.parquet").write_bytes(b"parquet content verified by load_instances at launch")
    config = {"models": pilot.MODELS, "parquet": "tasks.parquet", "reference_trace": "reference.json",
              "analysis_root": "analysis", "splits": {"train": ["train-1"], "dev": ["dev-1"], "pilot_eval": ["eval-1"]},
              "budget": {"max_candidate_proposals": 2, "reflection_batch_size": 1, "step_limit": 5, "seeds": [42, 43]}}
    path = tmp_path / "config.json"
    write_json(path, config)
    return path


@pytest.mark.parametrize("splits", [
    {"train": ["same"], "dev": ["same"], "pilot_eval": ["eval"]},
    {"train": ["same", "same"], "dev": ["dev"], "pilot_eval": ["eval"]},
    {"train": [], "dev": ["dev"], "pilot_eval": ["eval"]},
    {"train": ["../escape"], "dev": ["dev"], "pilot_eval": ["eval"]},
    {"train": ["train"], "dev": ["dev"], "test": ["test"]},
])
def test_invalid_or_overlapping_splits_rejected(splits):
    with pytest.raises(ValueError):
        pilot.validate_splits(splits)


def test_three_conditions_need_only_training_annotations(config_file, tmp_path):
    root = tmp_path / "bundle"
    manifest = pilot.prepare(config_file, root)
    shared = []
    for group in pilot.GROUPS:
        _, payloads, historical = pilot.load_bundle(root, group)
        shared.append(historical)
        assert set(payloads) == {"train-1"}
        text = json.dumps(payloads)
        assert "ANNOTATED_DIAGNOSIS" not in text
        if group == "baseline":
            assert payloads["train-1"]["provided_information"] == {}
        else:
            key = pilot.position.ANCHORS[pilot.GROUPS.index(group) - 1]
            assert payloads["train-1"]["provided_information"][key]["step"] == pilot.GROUPS.index(group)
    assert shared[0] == shared[1] == shared[2]
    assert "ORIGINAL_EDIT" in json.dumps(shared[0]) and "ORIGINAL_SUBMIT" in json.dumps(shared[0])
    assert "SECRET_CONFIG" not in json.dumps(shared[0]) and "SECRET_EXTRA" not in json.dumps(shared[0])
    assert manifest["anchor_statistics"]["inevitable_at_last_step"] == 1
    assert manifest["anchor_statistics"]["review_status"] == {"train-1": "unreported"}
    assert not any("dev-1" in name or "eval-1" in name for name in manifest["source_sha256"])


@pytest.mark.parametrize("target", ["trace.json", "analysis/train-1.analysis.json", "reference.json", "tasks.parquet", "bundle/inputs.json"])
def test_frozen_source_changes_rejected(config_file, tmp_path, target):
    root = tmp_path / "bundle"
    pilot.prepare(config_file, root)
    with (tmp_path / target).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        pilot.load_bundle(root, "baseline")


def test_preparation_refuses_to_replace_bundle(config_file, tmp_path):
    root = tmp_path / "bundle"
    pilot.prepare(config_file, root)
    with pytest.raises(FileExistsError):
        pilot.prepare(config_file, root)


def test_out_of_range_anchor_rejected_before_outputs(config_file, tmp_path):
    analysis_path = tmp_path / "analysis/train-1.analysis.json"
    analysis = pilot.read_json(analysis_path)
    analysis["step_attribution"][pilot.position.ANCHORS[0]]["message_index"] = 3
    write_json(analysis_path, analysis)
    with pytest.raises(ValueError, match="mismatch"):
        pilot.prepare(config_file, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_budget_does_not_accept_repeated_seeds(config_file):
    config = pilot.read_json(config_file)
    budget = copy.deepcopy(config["budget"])
    budget["seeds"] = [42, 42]
    with pytest.raises(ValueError, match="seeds"):
        pilot.validate_budget(budget, 3)
