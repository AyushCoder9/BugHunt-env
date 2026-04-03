"""
BugHunt task definitions.

Each task contains:
  - buggy_functions : dict of { name -> source_code_string_with_bug }
  - tests           : list of TestCase (run against live namespace → bool)
  - max_operations  : operation budget

Graders are 100% deterministic: we exec the agent's code,
call the function with known inputs, compare to known expected output.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Safe builtins for exec'd agent code
# ---------------------------------------------------------------------------
SAFE_BUILTINS: Dict[str, Any] = {
    "len": len, "range": range, "sum": sum, "zip": zip,
    "sorted": sorted, "reversed": reversed,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "isinstance": isinstance, "enumerate": enumerate,
    "any": any, "all": all, "map": map, "filter": filter,
    "True": True, "False": False, "None": None,
    "print": print,
}


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    test_id: str
    description: str
    # run(namespace) → True = pass, False = fail
    run: Callable[[Dict[str, Any]], bool]
    failure_hint: str = ""     # shown to agent on failure


def _run(expr: str, expected: Any, ns: Dict[str, Any]) -> bool:
    """Evaluate expr in namespace; return True iff result == expected."""
    try:
        result = eval(expr, {"__builtins__": SAFE_BUILTINS}, ns)
        return result == expected
    except Exception:
        return False


def _run_approx(expr: str, expected: float, tol: float, ns: Dict[str, Any]) -> bool:
    """Like _run but with float tolerance."""
    try:
        result = eval(expr, {"__builtins__": SAFE_BUILTINS}, ns)
        return abs(float(result) - expected) < tol
    except Exception:
        return False


# ---------------------------------------------------------------------------
# EASY — stats_module  (1 bug, 4 tests)
# ---------------------------------------------------------------------------
#
# Bug: calculate_average divides by (len - 1) instead of len.
#      For a 1-element list this causes ZeroDivisionError;
#      for longer lists it returns a wrong value.
#
EASY_FUNCTIONS: Dict[str, str] = {
    "calculate_average": '''\
def calculate_average(numbers):
    """Return the arithmetic mean of a list of numbers.
    Returns 0 for an empty list.
    """
    if not numbers:
        return 0
    total = sum(numbers)
    return total / (len(numbers) - 1)   # BUG: off-by-one in divisor
''',

    "find_maximum": '''\
def find_maximum(numbers):
    """Return (max_value, index) for the largest element.
    Returns (None, -1) for an empty list.
    """
    if not numbers:
        return None, -1
    max_val = numbers[0]
    max_idx = 0
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
            max_idx = i
    return max_val, max_idx
''',
}

EASY_TESTS: List[TestCase] = [
    TestCase(
        test_id="E1",
        description="calculate_average([2, 4, 6]) should return 4.0",
        run=lambda ns: _run_approx("calculate_average([2, 4, 6])", 4.0, 0.001, ns),
        failure_hint="Check the divisor in calculate_average.",
    ),
    TestCase(
        test_id="E2",
        description="calculate_average([10, 20, 30, 40]) should return 25.0",
        run=lambda ns: _run_approx("calculate_average([10, 20, 30, 40])", 25.0, 0.001, ns),
        failure_hint="The divisor is wrong for lists with 4 elements.",
    ),
    TestCase(
        test_id="E3",
        description="calculate_average([7]) should return 7.0",
        run=lambda ns: _run_approx("calculate_average([7])", 7.0, 0.001, ns),
        failure_hint="Single-element list hits a divide-by-zero. Fix the divisor.",
    ),
    TestCase(
        test_id="E4",
        description="calculate_average([]) should return 0 (edge case — no bug here)",
        run=lambda ns: _run("calculate_average([])", 0, ns),
        failure_hint="Empty list should return 0 — this should already pass.",
    ),
    TestCase(
        test_id="E5",
        description="find_maximum([3, 1, 4, 1, 5, 9, 2]) should return (9, 5)",
        run=lambda ns: _run("find_maximum([3, 1, 4, 1, 5, 9, 2])", (9, 5), ns),
        failure_hint="find_maximum has no bug — if this fails something else is wrong.",
    ),
]


# ---------------------------------------------------------------------------
# MEDIUM — text_processor  (2 independent bugs, 7 tests)
# ---------------------------------------------------------------------------
#
# Bug 1: reverse_words returns words in original order (forgot [::-1]).
# Bug 2: truncate_text doesn't account for the 3 chars used by "...".
#
MEDIUM_FUNCTIONS: Dict[str, str] = {
    "reverse_words": '''\
def reverse_words(sentence):
    """Return the sentence with word order reversed.
    E.g. "hello world" -> "world hello"
    Returns empty string for blank input.
    """
    if not sentence.strip():
        return ""
    words = sentence.split()
    return " ".join(words)   # BUG: missing reversal — should be words[::-1]
''',

    "truncate_text": '''\
def truncate_text(text, max_length):
    """Shorten text to at most max_length characters total.
    If truncated, append "..." (counted in max_length).
    E.g. truncate_text("Hello World", 8) -> "Hello..."
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."   # BUG: ignores 3 chars used by "..."
''',

    "count_words": '''\
def count_words(text):
    """Return the number of words in text."""
    if not text.strip():
        return 0
    return len(text.split())
''',
}

MEDIUM_TESTS: List[TestCase] = [
    TestCase(
        test_id="M1",
        description='reverse_words("hello world") should return "world hello"',
        run=lambda ns: _run('reverse_words("hello world")', "world hello", ns),
        failure_hint="The join is correct but the word list is not reversed.",
    ),
    TestCase(
        test_id="M2",
        description='reverse_words("the quick brown fox") should return "fox brown quick the"',
        run=lambda ns: _run('reverse_words("the quick brown fox")', "fox brown quick the", ns),
        failure_hint="Reverse the words list before joining.",
    ),
    TestCase(
        test_id="M3",
        description='reverse_words("single") should return "single"',
        run=lambda ns: _run('reverse_words("single")', "single", ns),
        failure_hint="Single word reversed is still the same word.",
    ),
    TestCase(
        test_id="M4",
        description='reverse_words("") should return ""',
        run=lambda ns: _run('reverse_words("")', "", ns),
        failure_hint="Empty string edge case — should already pass.",
    ),
    TestCase(
        test_id="M5",
        description='truncate_text("Hello World", 8) should return "Hello..." (8 chars total)',
        run=lambda ns: _run('truncate_text("Hello World", 8)', "Hello...", ns),
        failure_hint="The '...' takes 3 characters. Slice should be text[:max_length - 3].",
    ),
    TestCase(
        test_id="M6",
        description='truncate_text("Testing a long string", 10) should return "Testing..." (10 chars)',
        run=lambda ns: _run('truncate_text("Testing a long string", 10)', "Testing...", ns),
        failure_hint="Adjust the slice to leave room for '...'.",
    ),
    TestCase(
        test_id="M7",
        description='truncate_text("Hi", 10) should return "Hi" (no truncation needed)',
        run=lambda ns: _run('truncate_text("Hi", 10)', "Hi", ns),
        failure_hint="Short text should pass through unchanged — should already pass.",
    ),
]


# ---------------------------------------------------------------------------
# HARD — grade_calculator  (3 bugs, 2 interdependent, 9 tests)
# ---------------------------------------------------------------------------
#
# Bug 1 (weighted_average):  uses score + weight instead of score * weight.
#         This makes ALL weighted_average calls return garbage.
#
# Bug 2 (calculate_final_grade): midterm and final scores are swapped in the
#         call to weighted_average. This bug is MASKED by Bug 1 — it only
#         becomes visible after Bug 1 is fixed. Fixing Bug 2 alone changes
#         nothing observable. Agent must understand both to fix correctly.
#
# Bug 3 (class_statistics): uses > 60 instead of >= 60 for passing threshold.
#         Independent of the other two.
#
# Interdependency: agent fixing Bug 2 without Bug 1 → no improvement.
#                  agent fixing Bug 1 without Bug 2 → partial fix but wrong
#                  grades when midterm ≠ final.
#                  Both must be fixed together for all grade tests to pass.
#
HARD_FUNCTIONS: Dict[str, str] = {
    "letter_grade": '''\
def letter_grade(score):
    """Convert a numeric score to a letter grade.
    90-100 -> A, 80-89 -> B, 70-79 -> C, 60-69 -> D, below 60 -> F
    """
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
''',

    "weighted_average": '''\
def weighted_average(scores, weights):
    """Return the weighted average of scores.
    scores : list of numeric values
    weights: list of floats that sum to 1.0
    """
    if len(scores) != len(weights):
        return 0
    total = 0
    for score, weight in zip(scores, weights):
        total += score + weight   # BUG 1: should be score * weight
    return total
''',

    "calculate_final_grade": '''\
def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3, 0.3, 0.4)):
    """Compute the final letter grade for a student.
    assignments : list of assignment scores (0-100)
    midterm     : midterm score (0-100)
    final_exam  : final exam score (0-100)
    weights     : (assignment_weight, midterm_weight, final_weight) summing to 1.0
    """
    if not assignments:
        return "F"
    avg_assignment = sum(assignments) / len(assignments)
    composite = weighted_average(
        [avg_assignment, final_exam, midterm],   # BUG 2: final_exam and midterm are swapped
        list(weights)
    )
    return letter_grade(composite)
''',

    "class_statistics": '''\
def class_statistics(student_scores):
    """Compute summary statistics for a class.
    student_scores: dict of { student_name: numeric_score }
    Returns {"mean": float, "highest": float, "lowest": float, "passing": int}
    A student passes with a score >= 60.
    """
    if not student_scores:
        return {"mean": 0, "highest": 0, "lowest": 0, "passing": 0}
    scores = list(student_scores.values())
    passing = sum(1 for s in scores if s > 60)   # BUG 3: should be >= 60
    return {
        "mean": round(sum(scores) / len(scores), 2),
        "highest": max(scores),
        "lowest": min(scores),
        "passing": passing,
    }
''',
}

HARD_TESTS: List[TestCase] = [
    # weighted_average tests (affected by Bug 1 only)
    TestCase(
        test_id="H1",
        description="weighted_average([80, 90, 70], [0.3, 0.3, 0.4]) should return 79.0",
        run=lambda ns: _run_approx(
            "weighted_average([80, 90, 70], [0.3, 0.3, 0.4])", 79.0, 0.01, ns),
        failure_hint="Check the operator inside the loop: score + weight should be score * weight.",
    ),
    TestCase(
        test_id="H2",
        description="weighted_average([100, 100], [0.5, 0.5]) should return 100.0",
        run=lambda ns: _run_approx(
            "weighted_average([100, 100], [0.5, 0.5])", 100.0, 0.01, ns),
        failure_hint="With equal weights summing to 1, weighted average of equal values should equal that value.",
    ),
    TestCase(
        test_id="H3",
        description="weighted_average([50, 60, 70], [0.25, 0.25, 0.5]) should return 62.5",
        run=lambda ns: _run_approx(
            "weighted_average([50, 60, 70], [0.25, 0.25, 0.5])", 62.5, 0.01, ns),
        failure_hint="50*0.25 + 60*0.25 + 70*0.5 = 12.5 + 15 + 35 = 62.5",
    ),

    # calculate_final_grade tests (affected by Bug 1 AND Bug 2 together)
    TestCase(
        test_id="H4",
        description="calculate_final_grade([80,85], midterm=75, final_exam=90) should return 'B'",
        run=lambda ns: _run(
            'calculate_final_grade([80, 85], 75, 90)', "B", ns),
        failure_hint=(
            "Two bugs affect this: (1) score+weight vs score*weight, "
            "(2) midterm and final_exam are swapped in the call to weighted_average."
        ),
    ),
    TestCase(
        test_id="H5",
        description="calculate_final_grade([95,100], midterm=92, final_exam=98) should return 'A'",
        run=lambda ns: _run(
            'calculate_final_grade([95, 100], 92, 98)', "A", ns),
        failure_hint="Fix both Bug 1 and Bug 2 — swapped arguments only matter after arithmetic is corrected.",
    ),
    TestCase(
        test_id="H6",
        description="calculate_final_grade([55,60], midterm=50, final_exam=45) should return 'F'",
        run=lambda ns: _run(
            'calculate_final_grade([55, 60], 50, 45)', "F", ns),
        failure_hint="Low scores across the board should yield F.",
    ),
    TestCase(
        test_id="H7",
        description="calculate_final_grade([], midterm=80, final_exam=90) should return 'F' (no assignments)",
        run=lambda ns: _run(
            'calculate_final_grade([], 80, 90)', "F", ns),
        failure_hint="Empty assignments list returns F immediately — should already pass.",
    ),

    # class_statistics tests (affected by Bug 3 only — independent)
    TestCase(
        test_id="H8",
        description='class_statistics({"A":60,"B":75,"C":55})["passing"] should return 2',
        run=lambda ns: _run(
            'class_statistics({"A": 60, "B": 75, "C": 55})["passing"]', 2, ns),
        failure_hint="Score of exactly 60 should count as passing (>= 60, not > 60).",
    ),
    TestCase(
        test_id="H9",
        description='class_statistics({"A":85,"B":55})["passing"] should return 1',
        run=lambda ns: _run(
            'class_statistics({"A": 85, "B": 55})["passing"]', 1, ns),
        failure_hint="55 is below 60, only one student passes — should already pass.",
    ),
]


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str
    description: str
    context: str
    buggy_functions: Dict[str, str]
    tests: List[TestCase]
    max_operations: int
    num_bugs: int


def get_easy_task() -> Task:
    return Task(
        task_id="easy",
        difficulty="easy",
        description="Fix 1 bug in a statistics module",
        context=(
            "You are debugging a Python statistics module used by a data analysis pipeline. "
            "The module exports calculate_average and find_maximum. "
            "Something is wrong with calculate_average — three out of five tests are failing. "
            "Inspect the function, locate the bug, and propose a fix."
        ),
        buggy_functions=dict(EASY_FUNCTIONS),
        tests=list(EASY_TESTS),
        max_operations=10,
        num_bugs=1,
    )


def get_medium_task() -> Task:
    return Task(
        task_id="medium",
        difficulty="medium",
        description="Fix 2 independent bugs in a text processing module",
        context=(
            "You are debugging a text processing utility library. "
            "It exports reverse_words, truncate_text, and count_words. "
            "Four out of seven tests are failing — two different functions have bugs. "
            "You need to find and fix both independently."
        ),
        buggy_functions=dict(MEDIUM_FUNCTIONS),
        tests=list(MEDIUM_TESTS),
        max_operations=15,
        num_bugs=2,
    )


def get_hard_task() -> Task:
    return Task(
        task_id="hard",
        difficulty="hard",
        description="Fix 3 bugs in a grade calculator — two are interdependent",
        context=(
            "You are debugging a university grade calculation system. "
            "It exports letter_grade, weighted_average, calculate_final_grade, and class_statistics. "
            "Six out of nine tests are failing. "
            "WARNING: Two of the three bugs interact — fixing one without the other "
            "will NOT make the affected tests pass. You must understand both before either fix helps. "
            "The third bug is independent and can be fixed at any time."
        ),
        buggy_functions=dict(HARD_FUNCTIONS),
        tests=list(HARD_TESTS),
        max_operations=20,
        num_bugs=3,
    )


TASKS = {
    "easy":   get_easy_task,
    "medium": get_medium_task,
    "hard":   get_hard_task,
}
