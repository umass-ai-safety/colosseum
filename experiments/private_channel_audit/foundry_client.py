from __future__ import annotations

"""Azure Foundry API-key client compatible with the pinned Terrarium release.

Terrarium 0.1.1 predates its upstream Foundry provider.  This small adapter is kept
inside the new experiment rather than changing the shared framework dependency.
"""

import os
from typing import Any, Dict


def _foundry_base_url(project_endpoint: str) -> str:
    endpoint = str(project_endpoint or "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("Missing Azure Foundry project endpoint.")
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/"
    return f"{endpoint}/openai/v1/"


class FoundryOpenAIClient:
    """Build the existing OpenAI-compatible client against a Foundry project URL."""

    @staticmethod
    def from_config(foundry_cfg: Dict[str, Any]) -> Any:
        try:
            from llm_server.clients.openai_client import OpenAIClient
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "This experiment requires the OpenAI provider dependency already used by "
                "Terrarium (terrarium-agents[providers])."
            ) from exc

        cfg = foundry_cfg if isinstance(foundry_cfg, dict) else {}
        api_style = str(cfg.get("api_style") or "responses").strip().lower()
        if api_style != "responses":
            raise ValueError(
                "The private-channel audit experiment requires Foundry's Responses API."
            )
        auth_mode = str(cfg.get("auth_mode") or "api_key").strip().lower()
        if auth_mode != "api_key":
            raise ValueError(
                "The private-channel audit experiment supports only Foundry API-key authentication."
            )
        auth_mode_env = str(cfg.get("auth_mode_env_var") or "AI_FOUNDRY_AUTH_MODE")
        configured_auth_mode = os.getenv(auth_mode_env)
        if configured_auth_mode and configured_auth_mode.strip().lower() != auth_mode:
            raise ValueError(
                f"{auth_mode_env}={configured_auth_mode!r} conflicts with the "
                f"experiment's configured auth_mode={auth_mode!r}."
            )

        endpoint_env = str(
            cfg.get("project_endpoint_env_var") or "AI_FOUNDRY_PROJECT_ENDPOINT"
        )
        key_env = str(cfg.get("api_key_env_var") or "AI_FOUNDRY_API_KEY")
        endpoint = os.getenv(endpoint_env)
        api_key = os.getenv(key_env)
        if not endpoint:
            raise RuntimeError(f"Missing {endpoint_env}; set it before running this experiment.")
        if not api_key:
            raise RuntimeError(f"Missing {key_env}; set it before running this experiment.")

        # The pinned OpenAIClient has the complete Responses/tool-call handling needed
        # by BaseAgent. Construct it without its OpenAI-key-only initializer, then set
        # the same fields it uses for requests and retry behavior.
        client = OpenAIClient.__new__(OpenAIClient)
        client.api_key = api_key
        client.client = OpenAI(
            api_key=api_key,
            base_url=_foundry_base_url(endpoint),
            timeout=float(cfg.get("request_timeout", 60.0)),
        )
        client._max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
        client._backoff_initial_s = float(os.getenv("OPENAI_RETRY_INITIAL_S", "0.5"))
        client._backoff_max_s = float(os.getenv("OPENAI_RETRY_MAX_S", "20"))
        client._backoff_multiplier = float(
            os.getenv("OPENAI_RETRY_MULTIPLIER", "2")
        )
        client._backoff_jitter = float(os.getenv("OPENAI_RETRY_JITTER", "0.1"))
        return client
