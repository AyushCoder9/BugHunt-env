# server/environment.py
"""
BugHunt RL Environment — OpenEnv-compliant implementation.

The agent debugs Python code by:
1. Inspecting function source code (free)
2. Running test cases to see failures (free)
3. Proposing fixed function implementations (scored)
4. Submitting when done

Inherits from openenv.core.env_server.interfaces.Environment.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from openenv.core.env_server.interfaces import Environment

import sys
import os
from pathlib import Path

# Ensure parent package is importable
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    from ..models import BugHuntAction, BugHuntObservation, BugHuntState
except ImportError:
    from models import BugHuntAction, BugHuntObservation, BugHuntState

try:
    from .tasks import TASKS, Task
except ImportError:
    from server.tasks import TASKS, Task


SAFE_BUILTINS: Dict[str, Any] = {
    "len": len, "range": range, "sum": sum, "zip": zip,
    "sorted": sorted, "reversed": reversed, "min": min, "max": max,
    "abs": abs, "round": round, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "isinstance": isinstance,
    "enumerate": enumerate, "any": any, "all": all,
    "map": map, "filter": filter, "print": print,
    "True": True, "False": False, "None": None,
}

FORBIDDEN = [
    "__import__", "import os", "import sys", "import subprocess",
    "open(", "eval(", "exec(", "compile(", "globals(", "locals(",
    "getattr", "setattr", "delattr", "__class__",
]


class BugHuntEnvironment(Environment[BugHuntAction, BugHuntObservation, BugHuntState]):
    """
    BugHunt RL Environment.

    Reward:
    - inspect/run_test: 0.0 (free information gathering)
    - propose_fix that improves score: new_score - prev_score (> 0)
    - propose_fix with no improvement or invalid: -0.05
    - submit: final_score in [0.0, 1.0]
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = BugHuntState()
        self._task: Optional[Task] = None
        self._namespace: Dict[str, Any] = {}
        self._inspected: Dict[str, str] = {}
        self._test_results: Dict[str, dict] = {}
        self._ops_log: List[str] = []
        self._submitted = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> BugHuntObservation:
        """Reset the environment with a new task."""
        task_fn = TASKS.get(task_id, TASKS["easy"])
        self._task = task_fn()
        self._submitted = False
        self._ops_log = []
        self._inspected = {}

        # Shared namespace — all functions live here so cross-calls work
        self._namespace = {"__builtins__": SAFE_BUILTINS}
        for name, code in self._task.buggy_functions.items():
            try:
                exec(compile(code.strip(), f"<{name}>", "exec"), self._namespace)
            except Exception:
                pass

        self._test_results = {
            t.test_id: {
                "test_id": t.test_id,
                "description": t.description,
                "status": "not_run",
                "output": "",
            }
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
                f"Episode started. {self._task.max_operations} operations available. "
                f"Inspect functions, run tests, propose fixes, then submit."
            ),
        )

    def step(
        self,
        action: BugHuntAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BugHuntObservation:
        """Execute one action in the environment."""
        if self._task is None:
            # Environment not initialized — auto-reset to easy
            self.reset(task_id="easy")

        if self._submitted:
            return self._make_obs(
                reward=0.0, done=True, message="Episode already submitted."
            )

        self._state.step_count += 1

        if action.action_type == "submit":
            return self._handle_submit()
        elif action.action_type == "inspect_function":
            reward, message = self._handle_inspect(action)
        elif action.action_type == "run_test":
            reward, message = self._handle_run_test(action)
        elif action.action_type == "propose_fix":
            reward, message = self._handle_propose_fix(action)
        else:
            reward = -0.05
            message = (
                f"Unknown action '{action.action_type}'. "
                f"Use: inspect_function|run_test|propose_fix|submit"
            )

        ops_remaining = self._task.max_operations - self._state.step_count
        done = ops_remaining <= 0
        if done:
            self._submitted = True
            final = self._score()
            self._state.final_score = final
            self._state.is_submitted = True
            return self._make_obs(
                reward=final,
                done=True,
                message=f"Ops exhausted. Final score: {final:.2f}",
            )

        return self._make_obs(reward=reward, done=False, message=message)

    @property
    def state(self) -> BugHuntState:
        """Get the current environment state."""
        return self._state

    # ── Action handlers ──────────────────────────────────────────────────

    def _handle_inspect(self, action: BugHuntAction):
        name = action.function_name
        if not name or name not in self._task.buggy_functions:
            return (
                -0.05,
                f"No function '{name}'. Available: {list(self._task.buggy_functions.keys())}",
            )
        self._inspected[name] = self._task.buggy_functions[name]
        self._ops_log.append(f"inspect_function('{name}')")
        return 0.0, f"Inspected '{name}'. Source now in observation."

    def _handle_run_test(self, action: BugHuntAction):
        tid = action.test_id
        test_map = {t.test_id: t for t in self._task.tests}
        if not tid or tid not in test_map:
            return (
                -0.05,
                f"No test '{tid}'. Available: {list(test_map.keys())}",
            )
        test = test_map[tid]
        try:
            passed = test.run(self._namespace)
            status, output = ("pass", "") if passed else ("fail", test.failure_hint)
        except Exception as exc:
            status, output = "error", str(exc)
        self._test_results[tid] = {
            "test_id": tid,
            "description": test.description,
            "status": status,
            "output": output,
        }
        self._ops_log.append(f"run_test('{tid}') → {status}")
        verdict = "PASSED ✅" if status == "pass" else f"FAILED ❌ — {output}"
        return 0.0, f"Test {tid}: {verdict}"

    def _handle_propose_fix(self, action: BugHuntAction):
        name = action.function_name
        code = action.new_code
        if not name or not code:
            return -0.05, "propose_fix requires function_name and new_code."
        if name not in self._task.buggy_functions:
            return (
                -0.05,
                f"No function '{name}'. Available: {list(self._task.buggy_functions.keys())}",
            )
        for bad in FORBIDDEN:
            if bad in code:
                return -0.05, f"Forbidden pattern '{bad}' in code."
        if not code.strip().startswith("def "):
            return -0.05, "new_code must start with 'def '."

        prev_score = self._score()  # MUST be before exec

        try:
            exec(compile(code.strip(), f"<{name}>", "exec"), self._namespace)
        except SyntaxError as e:
            return -0.05, f"SyntaxError: {e}"
        except Exception as e:
            return -0.05, f"Error: {e}"

        # Re-run previously-run tests
        for tid, result in self._test_results.items():
            if result["status"] != "not_run":
                test = next(t for t in self._task.tests if t.test_id == tid)
                try:
                    passed = test.run(self._namespace)
                    self._test_results[tid] = {
                        "test_id": tid,
                        "description": test.description,
                        "status": "pass" if passed else "fail",
                        "output": "" if passed else test.failure_hint,
                    }
                except Exception as exc:
                    self._test_results[tid] = {
                        "test_id": tid,
                        "description": test.description,
                        "status": "error",
                        "output": str(exc),
                    }

        new_score = self._score()
        delta = new_score - prev_score
        reward = delta if delta > 0 else -0.05
        self._ops_log.append(
            f"propose_fix('{name}') score {prev_score:.2f}→{new_score:.2f}"
        )

        if delta > 0:
            return reward, f"Fix accepted! Score: {prev_score:.2f} → {new_score:.2f}"
        return (
            reward,
            f"Fix accepted but score didn't improve ({new_score:.2f}). Check your logic.",
        )

    def _handle_submit(self) -> BugHuntObservation:
        # Run ALL tests on submit
        for test in self._task.tests:
            try:
                passed = test.run(self._namespace)
                self._test_results[test.test_id] = {
                    "test_id": test.test_id,
                    "description": test.description,
                    "status": "pass" if passed else "fail",
                    "output": "" if passed else test.failure_hint,
                }
            except Exception as exc:
                self._test_results[test.test_id] = {
                    "test_id": test.test_id,
                    "description": test.description,
                    "status": "error",
                    "output": str(exc),
                }
        final = self._score()
        self._submitted = True
        self._state.final_score = final
        self._state.is_submitted = True
        passed_count = sum(
            1 for r in self._test_results.values() if r["status"] == "pass"
        )
        self._ops_log.append("submit()")
        return self._make_obs(
            reward=final,
            done=True,
            message=(
                f"Submitted! {passed_count}/{len(self._task.tests)} tests passing. "
                f"Score: {final:.2f}"
            ),
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _score(self) -> float:
        """Compute current score as fraction of tests passing."""
        if not self._task:
            return 0.0
        passed = 0
        for t in self._task.tests:
            try:
                if t.run(self._namespace):
                    passed += 1
            except Exception:
                pass
        return passed / len(self._task.tests)

    def _make_obs(self, reward, done, message) -> BugHuntObservation:
        """Construct an observation from current state."""
        if self._task is None:
            return BugHuntObservation(
                done=done, reward=reward, task_id="",
                task_description="No task loaded", task_context="Call reset first.",
                available_functions=[], available_tests=[],
                inspected_functions={}, test_results=[], operations_log=[],
                operations_remaining=0, current_score=0.0,
                tests_passed=0, tests_total=0, message=message,
            )
        ops_rem = max(0, self._task.max_operations - self._state.step_count)
        score = self._score()
        passed = sum(1 for t in self._task.tests if self._try_run(t))
        return BugHuntObservation(
            done=done,
            reward=reward,
            task_id=self._task.task_id,
            task_description=self._task.description,
            task_context=self._task.context,
            available_functions=list(self._task.buggy_functions.keys()),
            available_tests=[t.test_id for t in self._task.tests],
            inspected_functions=dict(self._inspected),
            test_results=list(self._test_results.values()),
            operations_log=list(self._ops_log),
            operations_remaining=ops_rem,
            current_score=score,
            tests_passed=passed,
            tests_total=len(self._task.tests),
            message=message,
        )

    def _try_run(self, test) -> bool:
        try:
            return bool(test.run(self._namespace))
        except Exception:
            return False
