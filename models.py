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
