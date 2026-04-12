# test_integration.py
"""
BugHunt v2 — Integration Tests

Tests environment logic, SDK compliance, and all 3 task difficulties.
Run with: .venv/bin/python -m pytest test_integration.py -v
"""
import sys
from pathlib import Path

# Ensure imports work
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from server.environment import BugHuntEnvironment
from models import BugHuntAction, BugHuntObservation, BugHuntState


def make_env():
    return BugHuntEnvironment()


# ── EASY ──────────────────────────────────────────────────────────────────────

def test_easy_initial_state():
    env = make_env()
    obs = env.reset(task_id="easy")
    assert obs.done is False
    assert obs.reward is None
    assert obs.tests_total == 5
    assert obs.operations_remaining == 10
    assert "calculate_average" in obs.available_functions
    assert 0.0 <= obs.current_score <= 1.0


def test_inspect_is_free():
    env = make_env()
    env.reset(task_id="easy")
    obs = env.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
    assert obs.reward == 0.0
    assert "calculate_average" in obs.inspected_functions
    assert "def calculate_average" in obs.inspected_functions["calculate_average"]


def test_run_test_is_free():
    env = make_env()
    env.reset(task_id="easy")
    obs = env.step(BugHuntAction(action_type="run_test", test_id="E1"))
    assert obs.reward == 0.0


def test_easy_correct_fix_gives_positive_reward():
    env = make_env()
    env.reset(task_id="easy")
    fix = (
        "def calculate_average(numbers):\n"
        "    if not numbers:\n"
        "        return 0\n"
        "    return sum(numbers) / len(numbers)\n"
    )
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
    assert obs.reward > 0
    assert obs.current_score == pytest.approx(1.0, abs=0.01)


def test_easy_full_solve():
    env = make_env()
    env.reset(task_id="easy")
    fix = (
        "def calculate_average(numbers):\n"
        "    if not numbers:\n"
        "        return 0\n"
        "    return sum(numbers) / len(numbers)\n"
    )
    env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
    obs = env.step(BugHuntAction(action_type="submit"))
    assert obs.done is True
    assert obs.reward == pytest.approx(1.0, abs=0.01)


# ── MEDIUM ─────────────────────────────────────────────────────────────────────

def test_medium_full_solve():
    env = make_env()
    env.reset(task_id="medium")

    fix_r = (
        "def reverse_words(sentence):\n"
        "    if not sentence.strip():\n"
        "        return ''\n"
        "    return ' '.join(sentence.split()[::-1])\n"
    )
    fix_t = (
        "def truncate_text(text, max_length):\n"
        "    if len(text) <= max_length:\n"
        "        return text\n"
        "    return text[:max_length-3] + '...'\n"
    )

    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="reverse_words", new_code=fix_r))
    assert obs.reward > 0

    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="truncate_text", new_code=fix_t))
    assert obs.reward > 0

    obs = env.step(BugHuntAction(action_type="submit"))
    assert obs.reward == pytest.approx(1.0, abs=0.01)


# ── HARD — INTERDEPENDENCY IS THE KEY TEST ─────────────────────────────────────

def test_hard_bug2_alone_does_not_help():
    """CRITICAL: Fixing Bug 2 alone must NOT improve score (masked by Bug 1)."""
    env = make_env()
    env.reset(task_id="hard")
    score_before = env._score()

    fix_order_only = (
        "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n"
        "    if not assignments:\n"
        "        return 'F'\n"
        "    avg = sum(assignments)/len(assignments)\n"
        "    composite = weighted_average([avg, midterm, final_exam], list(weights))\n"
        "    return letter_grade(composite)\n"
    )
    obs = env.step(BugHuntAction(
        action_type="propose_fix",
        function_name="calculate_final_grade",
        new_code=fix_order_only,
    ))
    assert obs.reward <= 0, f"Bug2 alone MUST NOT help — got reward={obs.reward}"
    assert obs.current_score == score_before, "Score must not change when only Bug2 is fixed"


def test_hard_full_solve():
    env = make_env()
    env.reset(task_id="hard")

    # Fix Bug 3 (independent — class_statistics)
    fix_stats = (
        "def class_statistics(student_scores):\n"
        "    if not student_scores:\n"
        "        return {'mean':0,'highest':0,'lowest':0,'passing':0}\n"
        "    scores = list(student_scores.values())\n"
        "    passing = sum(1 for s in scores if s >= 60)\n"
        "    return {'mean':round(sum(scores)/len(scores),2),'highest':max(scores),"
        "'lowest':min(scores),'passing':passing}\n"
    )
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="class_statistics", new_code=fix_stats))
    assert obs.reward > 0

    # Fix Bug 2 (swapped args — no improvement yet because Bug 1 still masks)
    fix_order = (
        "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n"
        "    if not assignments:\n"
        "        return 'F'\n"
        "    avg = sum(assignments)/len(assignments)\n"
        "    composite = weighted_average([avg, midterm, final_exam], list(weights))\n"
        "    return letter_grade(composite)\n"
    )
    env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix_order))

    # Fix Bug 1 (arithmetic — now both bugs fixed, grade tests pass)
    fix_weighted = (
        "def weighted_average(scores, weights):\n"
        "    if len(scores)!=len(weights):\n"
        "        return 0\n"
        "    total=0\n"
        "    for score,weight in zip(scores,weights):\n"
        "        total+=score*weight\n"
        "    return total\n"
    )
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix_weighted))
    assert obs.reward > 0

    obs = env.step(BugHuntAction(action_type="submit"))
    assert obs.reward == pytest.approx(1.0, abs=0.01)


# ── PENALTY TESTS ─────────────────────────────────────────────────────────────

def test_bad_function_name_penalty():
    env = make_env()
    env.reset(task_id="easy")
    obs = env.step(BugHuntAction(action_type="inspect_function", function_name="nonexistent"))
    assert obs.reward == -0.05


def test_bad_test_id_penalty():
    env = make_env()
    env.reset(task_id="easy")
    obs = env.step(BugHuntAction(action_type="run_test", test_id="ZZZ"))
    assert obs.reward == -0.05


def test_forbidden_import_penalty():
    env = make_env()
    env.reset(task_id="easy")
    bad_fix = "def calculate_average(numbers):\n    import os\n    return 0\n"
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=bad_fix))
    assert obs.reward == -0.05


def test_syntax_error_penalty():
    env = make_env()
    env.reset(task_id="easy")
    bad_fix = "def calculate_average(numbers):\n    return sum(numbers) /\n"
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=bad_fix))
    assert obs.reward == -0.05


def test_unknown_action_penalty():
    env = make_env()
    env.reset(task_id="easy")
    obs = env.step(BugHuntAction(action_type="fly_to_moon"))
    assert obs.reward == -0.05


def test_wrong_fix_no_improvement_penalty():
    env = make_env()
    env.reset(task_id="easy")
    wrong_fix = "def calculate_average(numbers):\n    return 999\n"
    obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=wrong_fix))
    assert obs.reward == -0.05


# ── SCORE RANGE TESTS ─────────────────────────────────────────────────────────

def test_score_always_in_range():
    for task_id in ["easy", "medium", "hard"]:
        env = make_env()
        env.reset(task_id=task_id)
        assert 0.0 <= env._score() <= 1.0
        obs = env.step(BugHuntAction(action_type="submit"))
        assert 0.0 <= obs.reward <= 1.0
        assert obs.done is True


# ── DETERMINISM TESTS ─────────────────────────────────────────────────────────

def test_graders_are_deterministic():
    def get_initial_score(task_id):
        env = make_env()
        env.reset(task_id=task_id)
        return env._score()

    for task_id in ["easy", "medium", "hard"]:
        scores = [get_initial_score(task_id) for _ in range(5)]
        assert len(set(scores)) == 1, f"{task_id} is not deterministic: {scores}"


# ── SDK COMPLIANCE TESTS ─────────────────────────────────────────────────────

def test_environment_inherits_from_openenv():
    from openenv.core.env_server.interfaces import Environment
    assert issubclass(BugHuntEnvironment, Environment)


def test_models_inherit_from_openenv_types():
    from openenv.core.env_server.types import Action, Observation, State
    assert issubclass(BugHuntAction, Action)
    assert issubclass(BugHuntObservation, Observation)
    assert issubclass(BugHuntState, State)


def test_observation_has_done_and_reward():
    env = make_env()
    obs = env.reset(task_id="easy")
    assert hasattr(obs, "done")
    assert hasattr(obs, "reward")


def test_state_has_episode_id_and_step_count():
    env = make_env()
    env.reset(task_id="easy")
    state = env.state
    assert hasattr(state, "episode_id")
    assert hasattr(state, "step_count")
    assert state.episode_id is not None
    assert state.step_count == 0


def test_step_increments_step_count():
    env = make_env()
    env.reset(task_id="easy")
    assert env.state.step_count == 0
    env.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
    assert env.state.step_count == 1


# ── APP IMPORT TEST ──────────────────────────────────────────────────────────

def test_app_is_fastapi():
    from server.app import app
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)
