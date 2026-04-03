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

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "easy",
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> BugHuntObservation:
        task_fn = TASKS.get(task_id, TASKS["easy"])
        self._task = task_fn()
        self._submitted = False
        self._ops_log = []
        self._inspected = {}

        # One shared namespace — all functions live here so cross-calls
        # (e.g. calculate_final_grade -> weighted_average) always resolve
        # to whatever implementation is currently installed.
        self._namespace = {"__builtins__": SAFE_BUILTINS}
        for name, code in self._task.buggy_functions.items():
            try:
                exec(compile(code.strip(), f"<{name}>", "exec"), self._namespace)
            except Exception:
                pass

        # All tests start as not_run
        self._test_results = {
            t.test_id: TestResult(
                test_id=t.test_id,
                description=t.description,
                status="not_run",
            )
            for t in self._task.tests
        }

        self._state = BugHuntState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
        )

        return self._make_obs(
            reward=None,
            done=False,
            message=(
                f"Episode started. You have {self._task.max_operations} operations. "
                f"Inspect functions, run tests, then propose fixes. Call submit when done."
            ),
        )

    def step(
        self,
        action: BugHuntAction,
        **kwargs,
    ) -> BugHuntObservation:
        if self._submitted:
            return self._make_obs(reward=0.0, done=True, message="Episode already submitted.")

        self._state.step_count += 1
