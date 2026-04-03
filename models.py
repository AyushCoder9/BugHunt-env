"""
Typed models for BugHunt — the debugging RL environment.
Follows the OpenEnv spec: Action, Observation, State as Pydantic models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BugHuntAction(BaseModel):
    """
    An action the agent takes to investigate and fix bugs.

    action_type options
    -------------------
    inspect_function   – read the source code of one function
    run_test           – run a specific test case, see pass/fail + output
    propose_fix        – replace a function with new implementation
    submit             – finalise the episode
    """
    action_type: str
    function_name: Optional[str] = None   # for inspect_function, propose_fix
    test_id: Optional[str] = None         # for run_test
    new_code: Optional[str] = None        # for propose_fix: full def block


class TestResult(BaseModel):
    test_id: str
    description: str
    status: str        # "pass" | "fail" | "error" | "not_run"
    output: str = ""   # what went wrong (empty on pass)


class BugHuntObservation(BaseModel):
    """What the agent sees after each action."""
    # OpenEnv base fields
    done: bool = False
    reward: Optional[float] = None

    # Task context
    task_id: str = ""
    task_description: str = ""
    task_context: str = ""          # narrative about what the module does

    # Available targets
