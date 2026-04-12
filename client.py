# client.py
"""
BugHunt Environment HTTP/WebSocket client.

Uses the OpenEnv EnvClient base class for standard async/sync access
to a running BugHunt server.
"""
from __future__ import annotations

from typing import Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import BugHuntAction, BugHuntObservation, BugHuntState
except ImportError:
    from models import BugHuntAction, BugHuntObservation, BugHuntState
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class BugHuntEnv(EnvClient[BugHuntAction, BugHuntObservation, BugHuntState]):
    """
    Client for the BugHunt debugging environment.

    Usage (async):
        >>> async with BugHuntEnv(base_url="http://localhost:7860") as env:
        ...     result = await env.reset(task_id="easy")
        ...     result = await env.step(BugHuntAction(
        ...         action_type="inspect_function",
        ...         function_name="calculate_average"
        ...     ))

    Usage (sync):
        >>> with BugHuntEnv(base_url="http://localhost:7860").sync() as env:
        ...     result = env.reset(task_id="easy")
        ...     result = env.step(BugHuntAction(
        ...         action_type="inspect_function",
        ...         function_name="calculate_average"
        ...     ))
    """

    def _step_payload(self, action: BugHuntAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BugHuntObservation]:
        obs_data = payload.get("observation", {})
        observation = BugHuntObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            task_context=obs_data.get("task_context", ""),
            available_functions=obs_data.get("available_functions", []),
            available_tests=obs_data.get("available_tests", []),
            inspected_functions=obs_data.get("inspected_functions", {}),
            test_results=obs_data.get("test_results", []),
            operations_log=obs_data.get("operations_log", []),
            operations_remaining=obs_data.get("operations_remaining", 0),
            current_score=obs_data.get("current_score", 0.0),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_total=obs_data.get("tests_total", 0),
            message=obs_data.get("message", ""),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> BugHuntState:
        return BugHuntState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            final_score=payload.get("final_score"),
            is_submitted=payload.get("is_submitted", False),
        )

    async def inspect_function(self, function_name: str) -> StepResult[BugHuntObservation]:
        """Convenience: inspect a function's source code."""
        return await self.step(BugHuntAction(
            action_type="inspect_function",
            function_name=function_name,
        ))

    async def run_test(self, test_id: str) -> StepResult[BugHuntObservation]:
        """Convenience: run a specific test case."""
        return await self.step(BugHuntAction(
            action_type="run_test",
            test_id=test_id,
        ))

    async def propose_fix(self, function_name: str, new_code: str) -> StepResult[BugHuntObservation]:
        """Convenience: propose a fix for a function."""
        return await self.step(BugHuntAction(
            action_type="propose_fix",
            function_name=function_name,
            new_code=new_code,
        ))

    async def submit(self) -> StepResult[BugHuntObservation]:
        """Convenience: submit and finalize the episode."""
        return await self.step(BugHuntAction(action_type="submit"))
