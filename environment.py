"""
BugHunt Environment — core RL logic.
Implements the OpenEnv 3-method interface: reset() / step() / state.
"""
from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional

from models import (
    BugHuntAction,
    BugHuntObservation,
    BugHuntState,
    TestResult,
)
from tasks import TASKS, Task, SAFE_BUILTINS


# Patterns we refuse in agent-submitted code
_FORBIDDEN = [
    "__import__", "import os", "import sys", "import subprocess",
    "open(", "eval(", "exec(", "compile(", "globals(", "locals(",
    "getattr", "setattr", "delattr", "vars(", "__class__",
]


class BugHuntEnvironment:
    """
    The agent inspects functions, runs tests, and proposes fixes.

    Reward design
    -------------
    propose_fix that increases tests_passed:  reward = Δscore  (> 0)
    propose_fix with no improvement:          reward = -0.05
    inspect_function / run_test:              reward =  0.0  (free information)
    invalid action:                           reward = -0.05
    submit():                                 reward = final_score ∈ [0, 1]
    auto-submit when ops exhausted:           same as submit()
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = BugHuntState()
        self._task: Optional[Task] = None
        self._namespace: Dict[str, Any] = {}   # live function implementations
        self._inspected: Dict[str, str] = {}    # functions the agent has read
        self._test_results: Dict[str, TestResult] = {}
        self._ops_log: List[str] = []
        self._submitted = False

