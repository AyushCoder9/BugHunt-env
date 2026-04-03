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
        ops_remaining = self._task.max_operations - self._state.step_count

        if action.action_type == "submit":
            return self._handle_submit()

        if action.action_type == "inspect_function":
            reward, message = self._handle_inspect(action)
        elif action.action_type == "run_test":
            reward, message = self._handle_run_test(action)
        elif action.action_type == "propose_fix":
            reward, message = self._handle_propose_fix(action)
        else:
            reward = -0.05
            message = f"Unknown action_type '{action.action_type}'. Use inspect_function | run_test | propose_fix | submit."

        done = ops_remaining <= 0
        if done:
            self._submitted = True
            final_score = self._score()
            self._state.final_score = final_score
            self._state.is_submitted = True
            return self._make_obs(
                reward=final_score,
                done=True,
                message=f"Operations exhausted — auto-submitted. Final score: {final_score:.2f}",
            )

        return self._make_obs(reward=reward, done=False, message=message)

    @property
    def state(self) -> BugHuntState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_inspect(self, action: BugHuntAction):
        name = action.function_name
        if not name:
            return -0.05, "inspect_function requires a function_name."
        if name not in self._task.buggy_functions:
            available = list(self._task.buggy_functions.keys())
            return -0.05, f"No function '{name}'. Available: {available}"
        self._inspected[name] = self._task.buggy_functions[name]
        self._ops_log.append(f"inspect_function('{name}')")
        return 0.0, f"Inspected '{name}'. Source code is now in your observation."

    def _handle_run_test(self, action: BugHuntAction):
        tid = action.test_id
        if not tid:
            return -0.05, "run_test requires a test_id."
        test_map = {t.test_id: t for t in self._task.tests}
        if tid not in test_map:
            available = [t.test_id for t in self._task.tests]
            return -0.05, f"No test '{tid}'. Available: {available}"

        test = test_map[tid]
        try:
            passed = test.run(self._namespace)
            status = "pass" if passed else "fail"
            output = "" if passed else test.failure_hint
        except Exception as exc:
            status = "error"
            output = str(exc)

        self._test_results[tid] = TestResult(
            test_id=tid,
            description=test.description,
            status=status,
            output=output,
        )
        self._ops_log.append(f"run_test('{tid}') → {status}")
        verdict = "PASSED ✅" if status == "pass" else f"FAILED ❌ — {output}"
        return 0.0, f"Test {tid}: {verdict}"

    def _handle_propose_fix(self, action: BugHuntAction):
        name = action.function_name
        code = action.new_code

        if not name or not code:
            return -0.05, "propose_fix requires both function_name and new_code."
        if name not in self._task.buggy_functions:
            available = list(self._task.buggy_functions.keys())
            return -0.05, f"No function '{name}'. Available: {available}"

        # Security check
        for bad in _FORBIDDEN:
            if bad in code:
                return -0.05, f"Forbidden pattern '{bad}' in proposed code."

        prev_score = self._score()

        ok, err = self._exec_into_namespace(code, name)
        if not ok:
            self._ops_log.append(f"propose_fix('{name}') -> REJECTED: {err}")
            return -0.05, f"Fix rejected — {err}" 

        # Re-run all previously-run tests so score updates immediately
        for tid, result in self._test_results.items():
            if result.status != "not_run":
                test = next(t for t in self._task.tests if t.test_id == tid)
                try:
                    passed = test.run(self._namespace)
                    self._test_results[tid] = TestResult(
                        test_id=tid,
                        description=test.description,
                        status="pass" if passed else "fail",
                        output="" if passed else test.failure_hint,
                    )
                except Exception as exc:
                    self._test_results[tid] = TestResult(
                        test_id=tid,
                        description=test.description,
                        status="error",
                        output=str(exc),
                    )

        new_score = self._score()
        delta = new_score - prev_score
        reward = delta if delta > 0 else -0.05

        self._ops_log.append(
            f"propose_fix('{name}') → accepted | score {prev_score:.2f} → {new_score:.2f}"
        )
        if delta > 0:
            return reward, f"Fix accepted! Score improved {prev_score:.2f} → {new_score:.2f}."
        else:
            return reward, f"Fix accepted but score did not improve ({new_score:.2f}). Check your logic."

    def _handle_submit(self) -> BugHuntObservation:
        # Run ALL tests before final score
        for test in self._task.tests:
            try:
                passed = test.run(self._namespace)
                status = "pass" if passed else "fail"
                output = "" if passed else test.failure_hint
            except Exception as exc:
                status = "error"
                output = str(exc)
            self._test_results[test.test_id] = TestResult(
                test_id=test.test_id,
                description=test.description,
                status=status,
                output=output,
            )

        final_score = self._score()
        self._submitted = True
        self._state.final_score = final_score
        self._state.is_submitted = True

        passed_count = sum(1 for r in self._test_results.values() if r.status == "pass")
        total = len(self._task.tests)
        self._ops_log.append("submit()")

        return self._make_obs(
            reward=final_score,
            done=True,
            message=(
                f"Submitted! Score: {final_score:.2f} "
                f"({passed_count}/{total} tests passing)."
            ),
        )

    # ------------------------------------------------------------------
    # Scoring & helpers
    # ------------------------------------------------------------------

    def _score(self) -> float:
        """Run ALL tests silently and return fraction passing."""
        if not self._task:
            return 0.0
        passed = sum(1 for t in self._task.tests if self._run_silent(t))
        return passed / len(self._task.tests)

    def _run_silent(self, test) -> bool:
        try:
            return bool(test.run(self._namespace))
        except Exception:
            return False

    def _exec_into_namespace(self, code: str, name: str):
        """
        Exec a function definition into the shared namespace.
        Returns (success: bool, error_msg: str).
        """
        code = code.strip()
        if not code.startswith("def "):
            return False, "Code must start with 'def '."
