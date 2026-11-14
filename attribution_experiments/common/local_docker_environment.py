"""Native local Docker with cached images and retained experiment containers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid

from .remote_runtime import RemoteInfrastructureError
from .udocker_environment import UdockerEnvironment, UdockerEnvironmentConfig


class LocalDockerEnvironment(UdockerEnvironment):
    def __init__(self, *, logger=None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)
        self.config = UdockerEnvironmentConfig(**kwargs)
        self.container_name = "rq1-" + uuid.uuid4().hex
        self.scratch_volumes = {}
        self.closed = False
        inspected = subprocess.run(["docker", "image", "inspect", self.config.image],
                                   check=True, capture_output=True, text=True, timeout=30)
        self.image_digest = json.loads(inspected.stdout)[0]["Id"]
        command = ["docker", "run", "--detach", "--pull=never", "--name", self.container_name,
                   "--label", "attribution.rq1.retained=true", "--workdir", self.config.cwd,
                   "--memory", os.environ.get("EXPERIMENT_LOCAL_MEMORY", "2g"),
                   "--cpus", os.environ.get("EXPERIMENT_LOCAL_CPUS", "2"),
                   "--pids-limit", "512", "--entrypoint", "/bin/bash"]
        for host, target in self.config.volumes.items():
            command.extend(["--mount", f"type=bind,src={host},dst={target}"])
        command.extend([self.image_digest, "-c", "sleep 21600"])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)

    def _run_host(self, args, *, timeout, input_text=None, check=False):
        # Reuse UdockerEnvironment's action, observation and submission semantics.
        # Its run options use flags also accepted by docker exec.
        index = args.index(self.container_name)
        options = [item for item in args[1:index] if item != "--nobanner" and not item.startswith("--volume=")]
        command = ["docker", "exec", "--interactive", *options, self.container_name,
                   "timeout", "--signal=KILL", str(timeout), *args[index + 1:]]
        result = subprocess.run(command, input=input_text, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout + 15, check=check)
        state = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", self.container_name],
                               capture_output=True, text=True, timeout=30, check=True)
        status = json.loads(state.stdout)
        if not status["Running"] or status.get("OOMKilled"):
            raise RemoteInfrastructureError("Local experiment container stopped or exhausted memory")
        return result

    def execute(self, *args, **kwargs):
        result = super().execute(*args, **kwargs)
        if result.get("extra", {}).get("exception_type") in {"RemoteInfrastructureError", "CalledProcessError"}:
            raise RemoteInfrastructureError(result["exception_info"])
        return result

    def serialize(self):
        result = super().serialize()
        result["info"]["config"]["image_digest"] = self.image_digest
        result["info"]["config"]["container_retained"] = True
        return result

    def cleanup(self):
        if not self.closed:
            subprocess.run(["docker", "stop", "--time", "5", self.container_name],
                           check=True, capture_output=True, text=True, timeout=30)
            self.closed = True
