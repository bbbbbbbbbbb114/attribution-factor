import json
from pathlib import Path
import subprocess

import pytest

from attribution_experiments.common.local_docker_environment import LocalDockerEnvironment
from attribution_experiments.common.remote_runtime import RemoteInfrastructureError


@pytest.fixture
def docker_calls(monkeypatch):
    commands = []
    state = {"Running": True, "OOMKilled": False}

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if command[:3] == ["docker", "image", "inspect"]:
            output = json.dumps([{"Id": "sha256:fixed-image"}])
        elif command[:2] == ["docker", "inspect"]:
            output = json.dumps(state)
        else:
            output = "command output"
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(subprocess, "run", run)
    env = LocalDockerEnvironment(image="official:latest", repo=Path("/unused"), executable=Path("/unused"))
    return env, commands, state


def test_native_execution_forwards_stdin_bounds_commands_and_retains_container(docker_calls):
    env, commands, _ = docker_calls
    result = env.execute({"command": "git apply -"}, input_text="line 1\nline 2\n", timeout=15)
    assert result["returncode"] == 0
    executed = next((command, kwargs) for command, kwargs in commands if command[:2] == ["docker", "exec"])
    assert executed[1]["input"] == "line 1\nline 2\n"
    assert executed[0][-6:] == ["timeout", "--signal=KILL", "15", "bash", "-c", "git apply -"]
    env.cleanup()
    env.cleanup()
    assert sum(command[:2] == ["docker", "stop"] for command, _ in commands) == 1
    assert not any(command[:2] in (["docker", "rm"], ["docker", "pull"]) for command, _ in commands)
    assert "--pull=never" in commands[1][0]
    assert env.serialize()["info"]["config"]["image_digest"] == "sha256:fixed-image"


def test_container_failure_cannot_be_used_as_task_feedback(docker_calls):
    env, _, state = docker_calls
    state["Running"] = False
    with pytest.raises(RemoteInfrastructureError):
        env.execute({"command": "test-command"})
    env.cleanup()
