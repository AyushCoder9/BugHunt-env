#!/usr/bin/env python3
"""
BugHunt — RIGOROUS Judge-Level Test Suite

Tests the RL environment the way a hackathon judge would:
- Full RL loop correctness (reset → explore → fix → submit)
- Reward signal quality (does it guide learning?)
- Edge cases that break environments
- State consistency across steps
- Determinism verification
- Hard mode interdependency proof
- Sandbox security
- Operation budget enforcement
- Multiple episode resets
- Partial fix scenarios
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from server.environment import BugHuntEnvironment
from models import BugHuntAction, BugHuntObservation, BugHuntState


def env():
    return BugHuntEnvironment()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FULL RL LOOP — Does the Gym API contract actually work?
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLLoop:
    """A judge would run a full RL training loop. Does it work?"""

    def test_reset_returns_valid_observation(self):
        e = env()
        obs = e.reset(task_id="easy")
        assert isinstance(obs, BugHuntObservation)
        assert obs.done is False
        assert obs.reward is None  # initial obs has no reward
        assert obs.tests_total > 0
        assert obs.operations_remaining > 0
        assert len(obs.available_functions) > 0
        assert len(obs.available_tests) > 0

    def test_step_returns_valid_observation(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert isinstance(obs, BugHuntObservation)
        assert obs.done is False
        assert obs.reward is not None  # step always returns a reward

    def test_full_episode_easy(self):
        """Complete episode: reset → inspect → test → fix → submit."""
        e = env()
        obs = e.reset(task_id="easy")
        assert obs.done is False

        # Step 1: Inspect
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert obs.reward == 0.0
        assert "calculate_average" in obs.inspected_functions

        # Step 2: Run a failing test
        obs = e.step(BugHuntAction(action_type="run_test", test_id="E1"))
        assert obs.reward == 0.0

        # Step 3: Fix
        fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
        assert obs.reward > 0  # score should improve

        # Step 4: Submit
        obs = e.step(BugHuntAction(action_type="submit"))
        assert obs.done is True
        assert obs.reward == pytest.approx(1.0, abs=0.01)

    def test_full_episode_medium(self):
        """Two bugs, both must be fixed."""
        e = env()
        e.reset(task_id="medium")

        fix1 = "def reverse_words(sentence):\n    if not sentence.strip():\n        return ''\n    return ' '.join(sentence.split()[::-1])\n"
        fix2 = "def truncate_text(text, max_length):\n    if len(text) <= max_length:\n        return text\n    return text[:max_length-3] + '...'\n"

        e.step(BugHuntAction(action_type="propose_fix", function_name="reverse_words", new_code=fix1))
        e.step(BugHuntAction(action_type="propose_fix", function_name="truncate_text", new_code=fix2))
        obs = e.step(BugHuntAction(action_type="submit"))
        assert obs.done is True
        assert obs.reward == pytest.approx(1.0, abs=0.01)

    def test_full_episode_hard(self):
        """Three bugs, two interdependent."""
        e = env()
        e.reset(task_id="hard")

        fix_stats = "def class_statistics(student_scores):\n    if not student_scores:\n        return {'mean':0,'highest':0,'lowest':0,'passing':0}\n    scores = list(student_scores.values())\n    passing = sum(1 for s in scores if s >= 60)\n    return {'mean':round(sum(scores)/len(scores),2),'highest':max(scores),'lowest':min(scores),'passing':passing}\n"
        fix_weighted = "def weighted_average(scores, weights):\n    if len(scores)!=len(weights):\n        return 0\n    total=0\n    for score,weight in zip(scores,weights):\n        total+=score*weight\n    return total\n"
        fix_grade = "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n    if not assignments:\n        return 'F'\n    avg = sum(assignments)/len(assignments)\n    composite = weighted_average([avg, midterm, final_exam], list(weights))\n    return letter_grade(composite)\n"

        e.step(BugHuntAction(action_type="propose_fix", function_name="class_statistics", new_code=fix_stats))
        e.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix_weighted))
        e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix_grade))
        obs = e.step(BugHuntAction(action_type="submit"))
        assert obs.done is True
        assert obs.reward == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REWARD SIGNAL QUALITY — Is the reward actually useful for RL training?
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewardSignal:
    """A judge evaluates whether rewards guide learning correctly."""

    def test_information_gathering_is_free(self):
        """Inspect and test MUST be free — agents should explore without cost."""
        e = env()
        e.reset(task_id="easy")
        for fn in ["calculate_average", "find_maximum"]:
            obs = e.step(BugHuntAction(action_type="inspect_function", function_name=fn))
            assert obs.reward == 0.0, f"Inspecting {fn} should be free"
        for tid in ["E1", "E2", "E3", "E4", "E5"]:
            obs = e.step(BugHuntAction(action_type="run_test", test_id=tid))
            assert obs.reward == 0.0, f"Running test {tid} should be free"

    def test_good_fix_positive_reward(self):
        """A correct fix MUST give positive reward."""
        e = env()
        e.reset(task_id="easy")
        fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
        assert obs.reward > 0, "Correct fix must give positive reward"

    def test_bad_fix_negative_reward(self):
        """A wrong fix MUST give negative reward."""
        e = env()
        e.reset(task_id="easy")
        bad = "def calculate_average(numbers):\n    return 42\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=bad))
        assert obs.reward == -0.05, "Bad fix must give -0.05 penalty"

    def test_same_fix_twice_no_reward(self):
        """Applying the same fix twice should NOT give reward the second time."""
        e = env()
        e.reset(task_id="easy")
        fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
        obs1 = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
        assert obs1.reward > 0
        obs2 = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
        assert obs2.reward == -0.05, "Re-applying same fix should NOT give positive reward"

    def test_reward_monotonicity(self):
        """Cumulative score should never decrease from correct fixes."""
        e = env()
        e.reset(task_id="medium")
        scores = [e._score()]
        
        fix1 = "def reverse_words(sentence):\n    if not sentence.strip():\n        return ''\n    return ' '.join(sentence.split()[::-1])\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="reverse_words", new_code=fix1))
        scores.append(e._score())

        fix2 = "def truncate_text(text, max_length):\n    if len(text) <= max_length:\n        return text\n    return text[:max_length-3] + '...'\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="truncate_text", new_code=fix2))
        scores.append(e._score())

        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1], f"Score dropped from {scores[i-1]} to {scores[i]}"

    def test_submit_reward_equals_final_score(self):
        """Submit reward must equal the fraction of tests passing."""
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="submit"))
        # Before any fix: E4 and E5 pass (2/5 = 0.4)
        assert obs.reward == pytest.approx(0.4, abs=0.01)

    def test_partial_fix_gives_partial_reward(self):
        """Fixing only one of two bugs on medium should give partial score."""
        e = env()
        e.reset(task_id="medium")
        fix1 = "def reverse_words(sentence):\n    if not sentence.strip():\n        return ''\n    return ' '.join(sentence.split()[::-1])\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="reverse_words", new_code=fix1))
        obs = e.step(BugHuntAction(action_type="submit"))
        assert 0.4 < obs.reward < 1.0, f"Partial fix should give partial score, got {obs.reward}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HARD MODE INTERDEPENDENCY — The crown jewel. Does masking actually work?
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterdependency:
    """THE critical test — bug masking must actually work."""

    def test_bug2_alone_zero_improvement(self):
        """CRITICAL: Fixing ONLY Bug 2 (swapped args) must NOT improve score."""
        e = env()
        e.reset(task_id="hard")
        score_before = e._score()

        fix_order = "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n    if not assignments:\n        return 'F'\n    avg = sum(assignments)/len(assignments)\n    composite = weighted_average([avg, midterm, final_exam], list(weights))\n    return letter_grade(composite)\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix_order))
        score_after = e._score()

        assert score_after == score_before, \
            f"Bug2 alone MUST NOT help! Score went from {score_before} to {score_after}"
        assert obs.reward <= 0, "Negative reward expected for masked fix"

    def test_bug1_alone_partial_improvement(self):
        """Fixing Bug 1 (arithmetic) alone should give SOME improvement."""
        e = env()
        e.reset(task_id="hard")
        score_before = e._score()

        fix_arith = "def weighted_average(scores, weights):\n    if len(scores)!=len(weights):\n        return 0\n    total=0\n    for score,weight in zip(scores,weights):\n        total+=score*weight\n    return total\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix_arith))
        score_after = e._score()

        assert score_after > score_before, \
            f"Bug1 alone SHOULD help somewhat. {score_before} → {score_after}"

    def test_bug1_plus_bug2_full_grade_solve(self):
        """Fixing Bug 1 + Bug 2 together should pass ALL grade-related tests."""
        e = env()
        e.reset(task_id="hard")

        fix1 = "def weighted_average(scores, weights):\n    if len(scores)!=len(weights):\n        return 0\n    total=0\n    for score,weight in zip(scores,weights):\n        total+=score*weight\n    return total\n"
        fix2 = "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n    if not assignments:\n        return 'F'\n    avg = sum(assignments)/len(assignments)\n    composite = weighted_average([avg, midterm, final_exam], list(weights))\n    return letter_grade(composite)\n"

        e.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix1))
        e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix2))

        # Run grade tests to verify
        for tid in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
            obs = e.step(BugHuntAction(action_type="run_test", test_id=tid))
            result = next(r for r in obs.test_results if r["test_id"] == tid)
            assert result["status"] == "pass", f"Test {tid} should pass after fixing Bug1+Bug2: {result}"

    def test_bug3_independent(self):
        """Bug 3 (class_statistics) is independent and should improve score alone."""
        e = env()
        e.reset(task_id="hard")
        score_before = e._score()

        fix3 = "def class_statistics(student_scores):\n    if not student_scores:\n        return {'mean':0,'highest':0,'lowest':0,'passing':0}\n    scores = list(student_scores.values())\n    passing = sum(1 for s in scores if s >= 60)\n    return {'mean':round(sum(scores)/len(scores),2),'highest':max(scores),'lowest':min(scores),'passing':passing}\n"
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="class_statistics", new_code=fix3))

        assert obs.reward > 0, "Independent Bug3 fix should improve score"
        assert e._score() > score_before

    def test_fix_order_doesnt_matter(self):
        """Fixing Bug2 first then Bug1 should yield same result as Bug1 then Bug2."""
        # Order A: Bug1 → Bug2
        e1 = env()
        e1.reset(task_id="hard", episode_id="order_a")
        fix1 = "def weighted_average(scores, weights):\n    if len(scores)!=len(weights):\n        return 0\n    total=0\n    for score,weight in zip(scores,weights):\n        total+=score*weight\n    return total\n"
        fix2 = "def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n    if not assignments:\n        return 'F'\n    avg = sum(assignments)/len(assignments)\n    composite = weighted_average([avg, midterm, final_exam], list(weights))\n    return letter_grade(composite)\n"
        e1.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix1))
        e1.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix2))
        score_a = e1._score()

        # Order B: Bug2 → Bug1
        e2 = env()
        e2.reset(task_id="hard", episode_id="order_b")
        e2.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix2))
        e2.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix1))
        score_b = e2._score()

        assert score_a == pytest.approx(score_b, abs=0.01), \
            f"Fix order shouldn't matter! Order A={score_a}, Order B={score_b}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDGE CASES — Things that break badly-built environments
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """A judge would try to break the environment."""

    def test_step_after_done(self):
        """Stepping after episode is done should return done=True harmlessly."""
        e = env()
        e.reset(task_id="easy")
        e.step(BugHuntAction(action_type="submit"))
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert obs.done is True
        assert obs.reward == 0.0

    def test_double_submit(self):
        """Submitting twice should not crash."""
        e = env()
        e.reset(task_id="easy")
        obs1 = e.step(BugHuntAction(action_type="submit"))
        obs2 = e.step(BugHuntAction(action_type="submit"))
        assert obs1.done is True
        assert obs2.done is True

    def test_reset_clears_state(self):
        """Reset must completely clear previous episode state."""
        e = env()
        e.reset(task_id="easy")
        fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
        assert e._score() == pytest.approx(1.0)

        # Reset to same task — should have bugs again
        obs = e.reset(task_id="easy")
        assert obs.current_score == pytest.approx(0.4, abs=0.01), "Reset didn't clear fixes!"
        assert len(obs.inspected_functions) == 0, "Reset didn't clear inspections!"
        assert len(obs.operations_log) == 0, "Reset didn't clear ops log!"

    def test_reset_to_different_task(self):
        """Reset to a different task should work cleanly."""
        e = env()
        e.reset(task_id="easy")
        obs = e.reset(task_id="hard")
        assert obs.tests_total == 9  # hard has 9 tests
        assert "weighted_average" in obs.available_functions

    def test_multiple_resets(self):
        """Multiple resets in a row should not leak state."""
        e = env()
        for _ in range(10):
            obs = e.reset(task_id="easy")
            assert obs.current_score == pytest.approx(0.4, abs=0.01)
            assert e.state.step_count == 0

    def test_inspect_nonexistent_function(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="does_not_exist"))
        assert obs.reward == -0.05

    def test_run_nonexistent_test(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="run_test", test_id="Z99"))
        assert obs.reward == -0.05

    def test_propose_fix_missing_function_name(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", new_code="def f():\n    pass\n"))
        assert obs.reward == -0.05

    def test_propose_fix_missing_code(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average"))
        assert obs.reward == -0.05

    def test_propose_fix_not_a_function(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code="x = 5"))
        assert obs.reward == -0.05

    def test_empty_action_type(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type=""))
        assert obs.reward == -0.05

    def test_unknown_task_id_falls_back(self):
        """Unknown task_id should fall back to easy, not crash."""
        e = env()
        obs = e.reset(task_id="impossible_difficulty")
        assert obs.tests_total == 5  # easy has 5 tests


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SANDBOX SECURITY — Can an agent escape the sandbox?
# ═══════════════════════════════════════════════════════════════════════════════

class TestSandbox:
    """Judges will try to inject malicious code."""

    def test_import_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    import os\n    return 0\n"))
        assert obs.reward == -0.05

    def test_eval_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    return eval('1+1')\n"))
        assert obs.reward == -0.05

    def test_exec_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    exec('x=1')\n    return 0\n"))
        assert obs.reward == -0.05

    def test_file_access_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    open('/etc/passwd')\n    return 0\n"))
        assert obs.reward == -0.05

    def test_dunder_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    return numbers.__class__.__bases__\n"))
        assert obs.reward == -0.05

    def test_subprocess_blocked(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    import subprocess\n    return 0\n"))
        assert obs.reward == -0.05

    def test_syntax_error_handled(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average",
                                    new_code="def calculate_average(numbers):\n    return !!!\n"))
        assert obs.reward == -0.05
        assert obs.done is False  # should NOT crash the episode


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATE CONSISTENCY — Does state track correctly?
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateConsistency:
    """State must be consistent throughout the episode."""

    def test_step_count_increments(self):
        e = env()
        e.reset(task_id="easy")
        for i in range(5):
            e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
            assert e.state.step_count == i + 1

    def test_operations_remaining_decrements(self):
        e = env()
        obs = e.reset(task_id="easy")
        initial = obs.operations_remaining
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert obs.operations_remaining == initial - 1

    def test_episode_id_persists(self):
        e = env()
        e.reset(task_id="easy", episode_id="my_episode_123")
        assert e.state.episode_id == "my_episode_123"
        e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert e.state.episode_id == "my_episode_123"

    def test_state_task_id_matches(self):
        e = env()
        e.reset(task_id="medium")
        assert e.state.task_id == "medium"

    def test_final_score_set_on_submit(self):
        e = env()
        e.reset(task_id="easy")
        assert e.state.final_score is None
        assert e.state.is_submitted is False
        e.step(BugHuntAction(action_type="submit"))
        assert e.state.final_score is not None
        assert e.state.is_submitted is True

    def test_ops_log_tracks_actions(self):
        e = env()
        e.reset(task_id="easy")
        e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        e.step(BugHuntAction(action_type="run_test", test_id="E1"))
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="find_maximum"))
        assert len(obs.operations_log) == 3

    def test_test_results_update_after_run(self):
        e = env()
        e.reset(task_id="easy")
        obs = e.step(BugHuntAction(action_type="run_test", test_id="E1"))
        e1_result = next(r for r in obs.test_results if r["test_id"] == "E1")
        assert e1_result["status"] in ("pass", "fail", "error")
        assert e1_result["status"] != "not_run"

    def test_inspected_functions_accumulate(self):
        e = env()
        e.reset(task_id="easy")
        e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        obs = e.step(BugHuntAction(action_type="inspect_function", function_name="find_maximum"))
        assert len(obs.inspected_functions) == 2
        assert "calculate_average" in obs.inspected_functions
        assert "find_maximum" in obs.inspected_functions


# ═══════════════════════════════════════════════════════════════════════════════
# 7. OPERATION BUDGET — Does the limit enforce correctly?
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperationBudget:
    """Ops budget must be enforced — auto-submit when exhausted."""

    def test_exhausting_ops_auto_submits(self):
        """When out of operations, episode should end automatically."""
        e = env()
        e.reset(task_id="easy")  # max_operations = 10
        for i in range(10):
            obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        # After 10 steps, obs.done should be True
        assert obs.done is True, "Episode should auto-submit when ops exhausted"
        assert obs.reward is not None

    def test_ops_remaining_reaches_zero(self):
        e = env()
        obs = e.reset(task_id="easy")
        max_ops = obs.operations_remaining
        for i in range(max_ops):
            obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
        assert obs.operations_remaining == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DETERMINISM — Same inputs → same outputs, always
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """RL environments MUST be deterministic for reproducible training."""

    def test_same_actions_same_rewards(self):
        """Identical action sequences must produce identical rewards."""
        def run_episode():
            e = env()
            e.reset(task_id="easy", episode_id="det_test")
            rewards = []
            fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
            obs = e.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
            rewards.append(obs.reward)
            obs = e.step(BugHuntAction(action_type="run_test", test_id="E1"))
            rewards.append(obs.reward)
            obs = e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
            rewards.append(obs.reward)
            obs = e.step(BugHuntAction(action_type="submit"))
            rewards.append(obs.reward)
            return rewards

        r1 = run_episode()
        r2 = run_episode()
        r3 = run_episode()
        assert r1 == r2 == r3, f"Non-deterministic! {r1} vs {r2} vs {r3}"

    def test_initial_scores_deterministic(self):
        for task_id in ["easy", "medium", "hard"]:
            scores = []
            for _ in range(5):
                e = env()
                e.reset(task_id=task_id)
                scores.append(e._score())
            assert len(set(scores)) == 1, f"{task_id} initial score varies: {scores}"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CROSS-FUNCTION INTERACTIONS — Fixes must not break other functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossFunctionInteractions:
    """Fixing one function must not break another."""

    def test_fixing_average_doesnt_break_maximum(self):
        e = env()
        e.reset(task_id="easy")
        # Run E5 (find_maximum test) first — should pass
        obs1 = e.step(BugHuntAction(action_type="run_test", test_id="E5"))
        r1 = next(r for r in obs1.test_results if r["test_id"] == "E5")
        assert r1["status"] == "pass"

        # Fix calculate_average
        fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))

        # E5 should STILL pass
        obs2 = e.step(BugHuntAction(action_type="run_test", test_id="E5"))
        r2 = next(r for r in obs2.test_results if r["test_id"] == "E5")
        assert r2["status"] == "pass", "Fixing calculate_average broke find_maximum!"

    def test_hard_cross_function_calls(self):
        """calculate_final_grade calls weighted_average which calls letter_grade."""
        e = env()
        e.reset(task_id="hard")
        # Fix weighted_average — this should NOT break letter_grade
        fix = "def weighted_average(scores, weights):\n    if len(scores)!=len(weights):\n        return 0\n    total=0\n    for score,weight in zip(scores,weights):\n        total+=score*weight\n    return total\n"
        e.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix))
        
        # letter_grade should still work (it has no bug)
        # H7 tests calculate_final_grade with empty assignments → returns "F" immediately
        obs = e.step(BugHuntAction(action_type="run_test", test_id="H7"))
        r = next(r for r in obs.test_results if r["test_id"] == "H7")
        assert r["status"] == "pass", "Fixing weighted_average broke letter_grade!"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SDK COMPLIANCE — Environment meets OpenEnv contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestSDKCompliance:
    """OpenEnv SDK contract requirements."""

    def test_environment_inherits_correctly(self):
        from openenv.core.env_server.interfaces import Environment
        e = env()
        assert isinstance(e, Environment)

    def test_action_inherits_correctly(self):
        from openenv.core.env_server.types import Action
        assert issubclass(BugHuntAction, Action)

    def test_observation_inherits_correctly(self):
        from openenv.core.env_server.types import Observation
        assert issubclass(BugHuntObservation, Observation)

    def test_state_inherits_correctly(self):
        from openenv.core.env_server.types import State
        assert issubclass(BugHuntState, State)

    def test_observation_has_done_reward_metadata(self):
        e = env()
        obs = e.reset(task_id="easy")
        assert hasattr(obs, "done")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "metadata")

    def test_state_has_episode_id_step_count(self):
        e = env()
        e.reset(task_id="easy")
        assert hasattr(e.state, "episode_id")
        assert hasattr(e.state, "step_count")

    def test_app_is_fastapi(self):
        from server.app import app
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)
