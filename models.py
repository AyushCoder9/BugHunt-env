# models.py
"""
Data models for the BugHunt Environment.

BugHunt is an RL environment where agents debug Python code by inspecting
functions, running tests, proposing fixes, and submitting solutions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class BugHuntAction(Action):
    """
    An action the agent takes to debug Python code.

    action_type options:
        inspect_function  - read source code of a function (free, reward=0)
        run_test          - run a test case, see pass/fail + hint (free, reward=0)
        propose_fix       - replace a function with corrected code (scored)
        submit            - finalise the episode
    """

    action_type: str = Field(
        ..., description="inspect_function|run_test|propose_fix|submit"
    )
    function_name: Optional[str] = Field(
        None, description="Target function name"
    )
    test_id: Optional[str] = Field(
        None, description="Test ID to run (e.g. E1, M3, H5)"
    )
    new_code: Optional[str] = Field(
        None, description="Complete def block for propose_fix"
    )


class BugHuntObservation(Observation):
    """Full observation returned after each action."""

    # Task info
    task_id: str = ""
    task_description: str = ""
    task_context: str = ""

    # What the agent can interact with
    available_functions: List[str] = Field(default_factory=list)
    available_tests: List[str] = Field(default_factory=list)

    # State the agent has built up
    inspected_functions: Dict[str, str] = Field(default_factory=dict)
    test_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Progress tracking
    operations_log: List[str] = Field(default_factory=list)
    operations_remaining: int = 20
    current_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    message: str = ""


class BugHuntState(State):
    """Episode metadata. episode_id and step_count are inherited from State."""

    task_id: str = ""
    final_score: Optional[float] = None
    is_submitted: bool = False


__all__ = ["BugHuntAction", "BugHuntObservation", "BugHuntState"]
