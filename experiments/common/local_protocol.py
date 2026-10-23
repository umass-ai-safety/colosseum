from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from terrarium.core.blackboard import Megaboard, format_blackboard_events_for_prompt
from terrarium.communication_protocols.base import BaseCommunicationProtocol
from terrarium.tools.environment import (
    EnvironmentToolsNotFoundError,
    instantiate_environment_tools,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolEvent:
    tool_name: str
    agent_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    phase: Optional[str]
    iteration: Optional[int]
    planning_round: Optional[int]


class LocalCommunicationProtocol(BaseCommunicationProtocol):
    """In-process protocol (no MCP server) backed by Megaboard + environment-specific tools."""

    def __init__(
        self, *, config: Dict[str, Any], megaboard: Optional[Megaboard] = None
    ):
        self.config = config
        self.megaboard = megaboard or Megaboard()
        self.environment = None
        self.current_planning_round: Optional[int] = None
        self.tool_events: List[ToolEvent] = []

        # Environment tools implementation (local, no server).
        self._env_tools: Any = None
        self._env_tools_env_name: Optional[str] = None
        self._env_tools_env: Any = None

    def _get_env_tools(self, environment: Any) -> Any:
        env_name = ""
        if environment is not None:
            # Allow environment subclasses to opt into a base toolset without requiring a
            # duplicate `*Tools` class to exist under terrarium/environments/**/*_tools.py.
            env_name = str(
                getattr(environment, "tools_environment_name", None)
                or environment.__class__.__name__
                or ""
            )
        if not env_name:
            raise EnvironmentToolsNotFoundError(
                "Environment missing or invalid (no __class__.__name__)."
            )
        # Tools hold a reference to the environment they mutate (Terrarium >= 0.2.0),
        # so rebuild them whenever the environment instance changes, not just its name.
        if (
            self._env_tools is None
            or self._env_tools_env_name != env_name
            or self._env_tools_env is not environment
        ):
            self._env_tools = instantiate_environment_tools(
                env_name, self.megaboard, environment=environment
            )
            self._env_tools_env_name = env_name
            self._env_tools_env = environment
        return self._env_tools

    def _record_tool_event(
        self,
        *,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        phase: Optional[str],
        iteration: Optional[int],
    ) -> None:
        self.tool_events.append(
            ToolEvent(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=dict(arguments or {}),
                result=dict(result or {}),
                phase=phase,
                iteration=iteration,
                planning_round=self.current_planning_round,
            )
        )

    def _scrub_secret_blackboards(self) -> None:
        """
        Remove tool-action trace events from secret blackboards.

        Some environment tool implementations call `blackboard_manager.log_action_to_blackboards(...)`,
        which logs actions to every blackboard an agent belongs to. For secret coalition channels,
        we want them to be used only for communication/scheming, not for action traces.
        """
        try:
            boards = getattr(self.megaboard, "blackboards", None)
            if not isinstance(boards, list):
                return
            for bb in boards:
                tmpl = getattr(bb, "template", None)
                if not isinstance(tmpl, dict):
                    continue
                is_secret = bool(tmpl.get("secret_channel")) or str(
                    tmpl.get("visibility") or ""
                ).lower() == "secret"
                if not is_secret:
                    continue
                logs = getattr(bb, "logs", None)
                if not isinstance(logs, list) or not logs:
                    continue
                bb.logs = [
                    e
                    for e in logs
                    if not (
                        isinstance(e, dict)
                        and str(e.get("kind") or "") == "action_executed"
                    )
                ]
        except Exception:
            return

    async def _prefetch_blackboard_events(
        self,
        agent_name: str,
        *,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, str]:
        blackboard_ids = self.megaboard.get_agent_blackboards(agent_name)

        def _sort_key(bb_id: str) -> int:
            try:
                return int(bb_id)
            except Exception:
                return 0

        contexts: Dict[str, str] = {}
        for bb_id_str in sorted(blackboard_ids, key=_sort_key):
            try:
                bb_id_int = int(bb_id_str)
            except Exception:
                continue

            # Prefetch is automatic; avoid polluting tool_events with synthetic get_blackboard_events calls.
            events = self.megaboard.get(bb_id_int, agent_name, limit=None)
            contexts[bb_id_str] = format_blackboard_events_for_prompt(
                events if isinstance(events, list) else []
            )

        return contexts

    async def environment_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        env = self.environment
        if env is None:
            result = {"error": "Environment not set on protocol."}
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=result,
                phase=phase,
                iteration=iteration,
            )
            return result

        try:
            env_tools = self._get_env_tools(env)
        except EnvironmentToolsNotFoundError as exc:
            result = {"error": str(exc)}
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=result,
                phase=phase,
                iteration=iteration,
            )
            return result

        # Terrarium >= 0.2.0 tools mutate the bound environment directly, so there is
        # no env_state snapshot to pass in and no state_updates to apply afterwards.
        # A tool can raise after mutating state; catch it here (with the same result
        # BaseAgent would build) so the call is still recorded and secret boards scrubbed.
        try:
            response = env_tools.handle_tool_call(
                tool_name,
                agent_name,
                arguments,
                phase=phase,
                iteration=iteration,
            )
        except Exception as exc:
            logger.exception("Environment tool %s raised", tool_name)
            response = {"error": f"Error executing {tool_name}: {exc}"}

        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=response if isinstance(response, dict) else {"result": response},
            phase=phase,
            iteration=iteration,
        )
        # Ensure secret channels remain communication-only (no action traces).
        self._scrub_secret_blackboards()
        return response

    async def blackboard_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = self.megaboard.handle_tool_call(
            tool_name,
            agent_name,
            arguments,
            phase=phase,
            iteration=iteration,
        )
        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=result,
            phase=phase,
            iteration=iteration,
        )
        # Be defensive: keep secret channels communication-only.
        self._scrub_secret_blackboards()
        return result

    async def get_all_blackboard_ids(self) -> List[str]:
        return self.megaboard.get_blackboard_string_ids()

    async def get_agent_blackboard_ids(self, agent_name: str) -> List[str]:
        return self.megaboard.get_agent_blackboards(agent_name)

    async def post_system_message(
        self,
        blackboard_id: int,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.megaboard.post_system_message(blackboard_id, kind, payload)

    async def generate_comm_network(
        self,
        participants: Sequence[str],
        context: str,
        template: Optional[Dict[str, Any]] = None,
    ) -> int:
        blackboard_id = self.megaboard.add_blackboard(list(participants), template)
        self.megaboard.post_system_message(
            blackboard_id,
            kind="context",
            payload={"message": context},
        )
        return blackboard_id

    async def agent_planning_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        planning_round: int,
    ) -> Dict[str, Any]:
        self.environment = environment
        self.current_planning_round = planning_round
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="planning",
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="planning",
            iteration=iteration,
            round_num=planning_round,
        )

    async def agent_execution_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
    ) -> Dict[str, Any]:
        self.environment = environment
        self.current_planning_round = None
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="execution",
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="execution",
            iteration=iteration,
            round_num=0,
        )

    async def agent_survey_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
    ) -> Dict[str, Any]:
        """Optional post-run survey turn to elicit beliefs without posting to blackboards."""
        self.environment = environment
        self.current_planning_round = None
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="survey",
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="survey",
            iteration=iteration,
            round_num=0,
        )
