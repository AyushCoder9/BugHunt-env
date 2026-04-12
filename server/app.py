# server/app.py
"""
FastAPI application entry point for the BugHunt environment.

Uses openenv.core.env_server.http_server.create_app to create all standard
OpenEnv endpoints: /health, /reset, /step, /state, /ws, /docs, /web

Additional custom endpoints:
  /analytics      — Episode metrics & aggregate statistics
  /curriculum     — Curriculum learning status
  /tasks/info     — Task metadata & bug dependency graph
  /leaderboard    — Top scores per task
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.responses import RedirectResponse

# Ensure imports work both as package and standalone
_SERVER_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SERVER_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

try:
    from ..models import BugHuntAction, BugHuntObservation
    from .environment import BugHuntEnvironment
    from .gradio_ui import build_bughunt_ui
except ImportError:
    from models import BugHuntAction, BugHuntObservation
    from server.environment import BugHuntEnvironment
    from server.gradio_ui import build_bughunt_ui


# ── Analytics store (in-memory, perfect for demo/hackathon) ─────────────────

class AnalyticsStore:
    """Track episode-level metrics for aggregate statistics."""

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []
        self.leaderboard: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._start_time = time.time()

    def record_episode(
        self,
        task_id: str,
        score: float,
        steps: int,
        duration_s: float,
        reward_trace: List[float],
        agent_id: str = "anonymous",
    ):
        entry = {
            "episode_id": str(uuid.uuid4())[:8],
            "task_id": task_id,
            "score": round(score, 4),
            "steps": steps,
            "duration_s": round(duration_s, 2),
            "reward_trace": reward_trace,
            "agent_id": agent_id,
            "timestamp": time.time(),
        }
        self.episodes.append(entry)

        # Update leaderboard (keep top 10 per task)
        self.leaderboard[task_id].append(entry)
        self.leaderboard[task_id].sort(key=lambda x: (-x["score"], x["steps"]))
        self.leaderboard[task_id] = self.leaderboard[task_id][:10]

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across all episodes."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "uptime_s": round(time.time() - self._start_time, 1),
                "per_task": {},
            }

        per_task = {}
        for task_id in ["easy", "medium", "hard"]:
            eps = [e for e in self.episodes if e["task_id"] == task_id]
            if eps:
                scores = [e["score"] for e in eps]
                steps_list = [e["steps"] for e in eps]
                durations = [e["duration_s"] for e in eps]
                per_task[task_id] = {
                    "episodes": len(eps),
                    "avg_score": round(sum(scores) / len(scores), 4),
                    "max_score": round(max(scores), 4),
                    "min_score": round(min(scores), 4),
                    "avg_steps": round(sum(steps_list) / len(steps_list), 1),
                    "avg_duration_s": round(sum(durations) / len(durations), 2),
                    "perfect_solves": sum(1 for s in scores if s >= 0.99),
                }

        return {
            "total_episodes": len(self.episodes),
            "uptime_s": round(time.time() - self._start_time, 1),
            "per_task": per_task,
        }


analytics = AnalyticsStore()


# ── Curriculum Learning Controller ──────────────────────────────────────────

class CurriculumController:
    """
    Automatic difficulty progression for RL training.

    Agents start on 'easy' and auto-promote when avg score > threshold.
    This implements curriculum learning — a key RL training technique.
    """

    PROGRESSION = ["easy", "medium", "hard"]
    PROMOTE_THRESHOLD = 0.8
    WINDOW_SIZE = 5

    def __init__(self):
        self.current_level = 0
        self.recent_scores: Dict[str, List[float]] = defaultdict(list)

    def get_current_task(self) -> str:
        return self.PROGRESSION[self.current_level]

    def record_score(self, task_id: str, score: float) -> Dict[str, Any]:
        self.recent_scores[task_id].append(score)
        # Keep window
        self.recent_scores[task_id] = self.recent_scores[task_id][-self.WINDOW_SIZE:]

        promoted = False
        current_task = self.PROGRESSION[self.current_level]

        if (
            task_id == current_task
            and len(self.recent_scores[task_id]) >= self.WINDOW_SIZE
            and sum(self.recent_scores[task_id]) / len(self.recent_scores[task_id])
            >= self.PROMOTE_THRESHOLD
            and self.current_level < len(self.PROGRESSION) - 1
        ):
            self.current_level += 1
            promoted = True

        return {
            "current_difficulty": self.PROGRESSION[self.current_level],
            "level": self.current_level,
            "promoted": promoted,
            "window_avg": round(
                sum(self.recent_scores.get(current_task, [0]))
                / max(1, len(self.recent_scores.get(current_task, [0]))),
                3,
            ),
            "promote_threshold": self.PROMOTE_THRESHOLD,
            "window_size": self.WINDOW_SIZE,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_difficulty": self.PROGRESSION[self.current_level],
            "level": self.current_level,
            "max_level": len(self.PROGRESSION) - 1,
            "recent_scores": dict(self.recent_scores),
            "promote_threshold": self.PROMOTE_THRESHOLD,
        }


curriculum = CurriculumController()


# ── Bug dependency graph metadata ──────────────────────────────────────────

BUG_DEPENDENCY_GRAPH = {
    "easy": {
        "bugs": [
            {
                "id": "E_BUG1",
                "function": "calculate_average",
                "type": "off_by_one",
                "description": "Divides by len(n)-1 instead of len(n)",
                "severity": "critical",
                "independent": True,
            }
        ],
        "dependencies": [],
        "fix_order": ["E_BUG1"],
    },
    "medium": {
        "bugs": [
            {
                "id": "M_BUG1",
                "function": "reverse_words",
                "type": "missing_operation",
                "description": "List not reversed before join",
                "severity": "major",
                "independent": True,
            },
            {
                "id": "M_BUG2",
                "function": "truncate_text",
                "type": "boundary_error",
                "description": "Slice ignores 3 chars used by '...'",
                "severity": "major",
                "independent": True,
            },
        ],
        "dependencies": [],
        "fix_order": ["M_BUG1", "M_BUG2"],
    },
    "hard": {
        "bugs": [
            {
                "id": "H_BUG1",
                "function": "weighted_average",
                "type": "operator_error",
                "description": "Uses + instead of * for weighted sum",
                "severity": "critical",
                "independent": False,
                "masks": ["H_BUG2"],
            },
            {
                "id": "H_BUG2",
                "function": "calculate_final_grade",
                "type": "argument_swap",
                "description": "midterm and final_exam arguments swapped",
                "severity": "critical",
                "independent": False,
                "masked_by": ["H_BUG1"],
            },
            {
                "id": "H_BUG3",
                "function": "class_statistics",
                "type": "comparison_error",
                "description": "> 60 should be >= 60",
                "severity": "minor",
                "independent": True,
            },
        ],
        "dependencies": [
            {
                "from": "H_BUG1",
                "to": "H_BUG2",
                "type": "masks",
                "description": (
                    "Bug 1 (wrong operator) masks Bug 2 (swapped args). "
                    "Fixing Bug 2 alone yields zero improvement because "
                    "the arithmetic is still wrong."
                ),
            }
        ],
        "fix_order": ["H_BUG3", "H_BUG1+H_BUG2"],
        "optimal_strategy": (
            "Fix H_BUG3 first (independent, quick win). "
            "Then fix H_BUG1 and H_BUG2 together in a single step."
        ),
    },
}


# ── Environment factory ────────────────────────────────────────────────────

def create_bughunt_environment() -> BugHuntEnvironment:
    """Factory function for BugHuntEnvironment instances."""
    return BugHuntEnvironment()


# ── Create app ──────────────────────────────────────────────────────────────

app = create_app(
    create_bughunt_environment,
    BugHuntAction,
    BugHuntObservation,
    env_name="bughunt_env",
    max_concurrent_envs=1,
    gradio_builder=build_bughunt_ui,
)

# ── Custom endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root — redirect to web UI if enabled, else docs."""
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
    if enable_web:
        return RedirectResponse(url="/web/")
    return RedirectResponse(url="/docs")


@app.get("/analytics")
def get_analytics():
    """
    📊 Aggregate episode analytics.

    Returns per-task statistics including average scores, step distributions,
    solve rates, and uptime. Useful for monitoring agent training progress.
    """
    return analytics.get_stats()


@app.post("/analytics/record")
def record_analytics(data: Dict[str, Any]):
    """
    Record a completed episode for analytics tracking.

    Body: {task_id, score, steps, duration_s, reward_trace?, agent_id?}
    """
    analytics.record_episode(
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0),
        steps=data.get("steps", 0),
        duration_s=data.get("duration_s", 0),
        reward_trace=data.get("reward_trace", []),
        agent_id=data.get("agent_id", "anonymous"),
    )
    return {"status": "recorded", "total_episodes": len(analytics.episodes)}


@app.get("/leaderboard")
def get_leaderboard():
    """
    🏆 Per-task leaderboard (top 10 scores).

    Returns the best episodes for each difficulty level,
    sorted by score (desc) then steps (asc).
    """
    return analytics.leaderboard


@app.get("/curriculum")
def get_curriculum():
    """
    📈 Curriculum learning status.

    Returns the current difficulty level, recent performance window,
    and promotion status. Agents auto-promote when avg score > 0.8.
    """
    return curriculum.get_status()


@app.post("/curriculum/step")
def curriculum_step(data: Dict[str, Any]):
    """
    Record a curriculum score and check for promotion.

    Body: {task_id, score}
    Returns: {current_difficulty, promoted, window_avg, ...}
    """
    return curriculum.record_score(
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0),
    )


@app.get("/tasks/info")
def get_tasks_info():
    """
    📋 Task metadata & bug dependency graphs.

    Returns detailed information about each task including:
    - Bug descriptions and types
    - Interdependency relationships (critical for hard mode)
    - Optimal fix ordering
    - Available functions and test counts
    """
    return {
        "tasks": {
            "easy": {
                "difficulty": "easy",
                "bugs": 1,
                "tests": 5,
                "max_operations": 10,
                "functions": ["calculate_average", "find_maximum"],
                "challenge": "Find the off-by-one error",
            },
            "medium": {
                "difficulty": "medium",
                "bugs": 2,
                "tests": 7,
                "max_operations": 15,
                "functions": ["reverse_words", "truncate_text", "count_words"],
                "challenge": "Two independent bugs in different functions",
            },
            "hard": {
                "difficulty": "hard",
                "bugs": 3,
                "tests": 9,
                "max_operations": 20,
                "functions": [
                    "letter_grade",
                    "weighted_average",
                    "calculate_final_grade",
                    "class_statistics",
                ],
                "challenge": "Two interdependent bugs — fixing one alone yields zero improvement",
            },
        },
        "bug_dependency_graph": BUG_DEPENDENCY_GRAPH,
    }


@app.get("/tasks/dependency_graph/{task_id}")
def get_dependency_graph(task_id: str):
    """
    🔗 Bug dependency graph for a specific task.

    The hard task's dependency graph is the unique differentiator of BugHunt:
    two bugs MASK each other, requiring agents to reason about coupling.
    """
    if task_id not in BUG_DEPENDENCY_GRAPH:
        return {"error": f"Unknown task '{task_id}'. Use: easy, medium, hard"}
    return BUG_DEPENDENCY_GRAPH[task_id]


@app.get("/env/capabilities")
def get_capabilities():
    """
    🔧 Environment capabilities and features.

    Returns what makes this environment unique for RL training.
    """
    return {
        "name": "BugHunt",
        "version": "2.0.0",
        "sdk": "openenv-core",
        "features": {
            "deterministic_grading": True,
            "curriculum_learning": True,
            "interdependent_bugs": True,
            "reward_shaping": True,
            "sandboxed_execution": True,
            "concurrent_sessions": True,
            "analytics_tracking": True,
            "leaderboard": True,
            "bug_dependency_graph": True,
            "web_ui": True,
        },
        "action_space": [
            "inspect_function",
            "run_test",
            "propose_fix",
            "submit",
        ],
        "observation_fields": [
            "task_id", "task_description", "task_context",
            "available_functions", "available_tests",
            "inspected_functions", "test_results",
            "operations_log", "operations_remaining",
            "current_score", "tests_passed", "tests_total",
            "message",
        ],
        "reward_design": {
            "inspect_function": 0.0,
            "run_test": 0.0,
            "propose_fix_improved": "new_score - old_score",
            "propose_fix_no_improvement": -0.05,
            "invalid_action": -0.05,
            "submit": "final_score in [0, 1]",
        },
        "difficulty_levels": {
            "easy": {"bugs": 1, "tests": 5, "max_ops": 10, "interdependent": False},
            "medium": {"bugs": 2, "tests": 7, "max_ops": 15, "interdependent": False},
            "hard": {"bugs": 3, "tests": 9, "max_ops": 20, "interdependent": True},
        },
    }


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the BugHunt environment server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()