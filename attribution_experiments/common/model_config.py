"""Model identities and OpenAI-compatible endpoints from environment variables."""

from __future__ import annotations

import os
import json
from pathlib import Path


OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "OPTIMIZER_MODEL")
OPTIMIZER_LITELLM_MODEL = f"openai/{OPTIMIZER_MODEL}"
OPTIMIZER_BASE_URL = os.environ.get("OPTIMIZER_BASE_URL", "http://localhost:8000/v1")

TASK_MODEL = os.environ.get("TASK_MODEL", "TASK_MODEL")
TASK_BASE_URL = os.environ.get(
    "TASK_MODEL_BASE_URL",
    "http://localhost:8001/v1",
)
TASK_API_KEY_FILE = Path(__file__).resolve().parents[1] / ".secrets" / "task_model.env"
TASK_API_KEY_SOURCE = "TASK_MODEL_API_KEY environment variable"
TASK_TEMPERATURE = 1.0
TASK_TOP_P = float(os.environ.get("TASK_MODEL_TOP_P", "0.95"))
TASK_STEP_LIMIT = 250


def optimizer_api_key() -> str:
    key = os.environ.get("OPTIMIZER_API_KEY")
    if not key:
        raise RuntimeError("OPTIMIZER_API_KEY is required for optimizer/evolve calls")
    return key


def task_api_key() -> str:
    key = os.environ.get("TASK_MODEL_API_KEY")
    if not key:
        raise RuntimeError("TASK_MODEL_API_KEY is required for task-agent rollouts")
    return key


def optimizer_extra_body() -> dict:
    body = json.loads(os.environ.get("OPTIMIZER_EXTRA_BODY", "{}"))
    if not isinstance(body, dict):
        raise ValueError("OPTIMIZER_EXTRA_BODY must be a JSON object")
    return body


def task_model_kwargs(*, api_key: str, temperature: float = TASK_TEMPERATURE) -> dict:
    """Return portable OpenAI-compatible settings and optional provider parameters."""
    extra_body = json.loads(os.environ.get("TASK_MODEL_EXTRA_BODY", "{}"))
    if not isinstance(extra_body, dict):
        raise ValueError("TASK_MODEL_EXTRA_BODY must be a JSON object")
    kwargs = {
        "api_key": api_key,
        "api_base": TASK_BASE_URL,
        "custom_llm_provider": "openai",
        "drop_params": True,
        "parallel_tool_calls": True,
        "temperature": temperature,
        "timeout": float(os.environ.get("TASK_MODEL_REQUEST_TIMEOUT_SECONDS", "300")),
        "top_p": TASK_TOP_P,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs
