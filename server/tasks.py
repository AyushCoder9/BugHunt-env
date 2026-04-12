# server/tasks.py
"""
BugHunt task definitions — three difficulty levels with deterministic graders.

Each task contains:
- Buggy function source code (what the agent sees)
- Deterministic test cases with failure hints
- Metadata (difficulty, max operations, num bugs)

Task design principles:
- Easy: 1 bug, 5 tests, 10 ops — straightforward off-by-one
- Medium: 2 independent bugs, 7 tests, 15 ops — two unrelated fixes
- Hard: 3 bugs (2 interdependent), 9 tests, 20 ops — must understand coupling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


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


@dataclass
class TestCase:
    test_id: str
    description: str
    run: Callable[[Dict[str, Any]], bool]
    failure_hint: str = ""


def _run(expr: str, expected: Any, ns: Dict[str, Any]) -> bool:
    """Evaluate expr in a safe namespace and compare to expected."""
    try:
        return eval(expr, {"__builtins__": SAFE_BUILTINS}, ns) == expected
    except Exception:
        return False


def _approx(expr: str, expected: float, tol: float, ns: Dict[str, Any]) -> bool:
    """Evaluate expr and check approximate equality."""
    try:
        return abs(float(eval(expr, {"__builtins__": SAFE_BUILTINS}, ns)) - expected) < tol
    except Exception:
        return False


# ── EASY: stats_module — 1 bug, 5 tests ─────────────────────────────────────
#
# Bug: calculate_average divides by (len-1) instead of len
#      ZeroDivisionError on single-element, wrong value on all others

EASY_FUNCTIONS = {
    "calculate_average": '''\
def calculate_average(numbers):
    """Return the arithmetic mean. Returns 0 for empty list."""
    if not numbers:
        return 0
    total = sum(numbers)
    return total / (len(numbers) - 1)   # BUG: off-by-one divisor
''',
    "find_maximum": '''\
def find_maximum(numbers):
    """Return (max_value, index). Returns (None, -1) for empty list."""
    if not numbers:
        return None, -1
    max_val, max_idx = numbers[0], 0
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val, max_idx = numbers[i], i
    return max_val, max_idx
''',
}

EASY_TESTS = [
    TestCase("E1", "calculate_average([2,4,6]) == 4.0",
             lambda ns: _approx("calculate_average([2,4,6])", 4.0, 0.001, ns),
             "Check the divisor — should be len(numbers) not len(numbers)-1."),
    TestCase("E2", "calculate_average([10,20,30,40]) == 25.0",
             lambda ns: _approx("calculate_average([10,20,30,40])", 25.0, 0.001, ns),
             "4-element list: divisor wrong."),
    TestCase("E3", "calculate_average([7]) == 7.0 (single element, ZeroDivisionError bug)",
             lambda ns: _approx("calculate_average([7])", 7.0, 0.001, ns),
             "Single element: len-1 = 0 causes ZeroDivisionError."),
    TestCase("E4", "calculate_average([]) == 0 (edge case, already correct)",
             lambda ns: _run("calculate_average([])", 0, ns),
             "Empty list should return 0."),
    TestCase("E5", "find_maximum([3,1,4,1,5,9,2]) == (9, 5) (no bug)",
             lambda ns: _run("find_maximum([3,1,4,1,5,9,2])", (9, 5), ns),
             "find_maximum has no bug."),
]


# ── MEDIUM: text_processor — 2 independent bugs, 7 tests ────────────────────
#
# Bug 1: reverse_words missing [::-1]
# Bug 2: truncate_text slices at max_length instead of max_length-3

MEDIUM_FUNCTIONS = {
    "reverse_words": '''\
def reverse_words(sentence):
    """Return sentence with word order reversed. Empty string -> empty string."""
    if not sentence.strip():
        return ""
    words = sentence.split()
    return " ".join(words)   # BUG: missing [::-1]
''',
    "truncate_text": '''\
def truncate_text(text, max_length):
    """Shorten text to max_length chars total (including "..."). """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."   # BUG: ignores 3 chars used by "..."
''',
    "count_words": '''\
def count_words(text):
    """Return number of words in text."""
    if not text.strip():
        return 0
    return len(text.split())
''',
}

MEDIUM_TESTS = [
    TestCase("M1", 'reverse_words("hello world") == "world hello"',
             lambda ns: _run('reverse_words("hello world")', "world hello", ns),
             "The split is correct but the list is not reversed before joining."),
    TestCase("M2", 'reverse_words("the quick brown fox") == "fox brown quick the"',
             lambda ns: _run('reverse_words("the quick brown fox")', "fox brown quick the", ns),
             "Use words[::-1] before joining."),
    TestCase("M3", 'reverse_words("single") == "single"',
             lambda ns: _run('reverse_words("single")', "single", ns),
             "Single word reversed is still the same."),
    TestCase("M4", 'reverse_words("") == ""',
             lambda ns: _run('reverse_words("")', "", ns),
             "Empty string edge case."),
    TestCase("M5", 'truncate_text("Hello World", 8) == "Hello..." (8 chars)',
             lambda ns: _run('truncate_text("Hello World", 8)', "Hello...", ns),
             'Slice should be text[:max_length-3] to leave room for "...".'),
    TestCase("M6", 'truncate_text("Testing a long string", 10) == "Testing..."',
             lambda ns: _run('truncate_text("Testing a long string", 10)', "Testing...", ns),
             "max_length=10, '...' is 3 chars, so slice at [:7]."),
    TestCase("M7", 'truncate_text("Hi", 10) == "Hi" (no truncation)',
             lambda ns: _run('truncate_text("Hi", 10)', "Hi", ns),
             "Short text passes through unchanged."),
]


# ── HARD: grade_calculator — 3 bugs, 2 interdependent, 9 tests ───────────────
#
# Bug 1 (weighted_average):    score + weight instead of score * weight
# Bug 2 (calculate_final_grade): midterm/final_exam SWAPPED — MASKED by Bug 1
# Bug 3 (class_statistics):    > 60 instead of >= 60 (independent)
#
# INTERDEPENDENCY:
#   Fixing Bug 2 alone → reward = -0.05 (zero observable improvement)
#   Fixing Bug 1 alone → partial fix (arithmetic right, but order still wrong)
#   Fixing Bug 1 + Bug 2 → all grade tests pass
#   Agent MUST understand both before fixing either

HARD_FUNCTIONS = {
    "letter_grade": '''\
def letter_grade(score):
    """90-100=A, 80-89=B, 70-79=C, 60-69=D, <60=F"""
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"
''',
    "weighted_average": '''\
def weighted_average(scores, weights):
    """Weighted average of scores. weights should sum to 1.0."""
    if len(scores) != len(weights):
        return 0
    total = 0
    for score, weight in zip(scores, weights):
        total += score + weight   # BUG 1: + instead of *
    return total
''',
    "calculate_final_grade": '''\
def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3, 0.3, 0.4)):
    """Compute letter grade. weights = (assignment, midterm, final) summing to 1.0"""
    if not assignments:
        return "F"
    avg_assignment = sum(assignments) / len(assignments)
    composite = weighted_average(
        [avg_assignment, final_exam, midterm],   # BUG 2: final_exam and midterm SWAPPED
        list(weights)
    )
    return letter_grade(composite)
''',
    "class_statistics": '''\
def class_statistics(student_scores):
    """Stats for a class. passing = score >= 60."""
    if not student_scores:
        return {"mean": 0, "highest": 0, "lowest": 0, "passing": 0}
    scores = list(student_scores.values())
    passing = sum(1 for s in scores if s > 60)   # BUG 3: > should be >=
    return {
        "mean": round(sum(scores) / len(scores), 2),
        "highest": max(scores),
        "lowest": min(scores),
        "passing": passing,
    }
''',
}

HARD_TESTS = [
    TestCase("H1", "weighted_average([80,90,70],[0.3,0.3,0.4]) == 79.0",
             lambda ns: _approx("weighted_average([80,90,70],[0.3,0.3,0.4])", 79.0, 0.01, ns),
             "The loop uses + instead of *. Fix: total += score * weight"),
    TestCase("H2", "weighted_average([100,100],[0.5,0.5]) == 100.0",
             lambda ns: _approx("weighted_average([100,100],[0.5,0.5])", 100.0, 0.01, ns),
             "100*0.5 + 100*0.5 = 100.0"),
    TestCase("H3", "weighted_average([50,60,70],[0.25,0.25,0.5]) == 62.5",
             lambda ns: _approx("weighted_average([50,60,70],[0.25,0.25,0.5])", 62.5, 0.01, ns),
             "50*0.25 + 60*0.25 + 70*0.5 = 62.5"),
    TestCase("H4", "calculate_final_grade([80,85],75,90) == 'B'",
             lambda ns: _run("calculate_final_grade([80,85],75,90)", "B", ns),
             "Two bugs interact: fix weighted_average (+→*) AND fix swapped midterm/final_exam."),
    TestCase("H5", "calculate_final_grade([95,100],92,98) == 'A'",
             lambda ns: _run("calculate_final_grade([95,100],92,98)", "A", ns),
             "Fix both Bug 1 and Bug 2. Bug 2 only matters after Bug 1 is fixed."),
    TestCase("H6", "calculate_final_grade([55,60],50,45) == 'F'",
             lambda ns: _run("calculate_final_grade([55,60],50,45)", "F", ns),
             "Low scores should yield F."),
    TestCase("H7", "calculate_final_grade([],80,90) == 'F' (no assignments)",
             lambda ns: _run("calculate_final_grade([],80,90)", "F", ns),
             "Empty assignments returns F immediately."),
    TestCase("H8", 'class_statistics({"A":60,"B":75,"C":55})["passing"] == 2',
             lambda ns: _run('class_statistics({"A":60,"B":75,"C":55})["passing"]', 2, ns),
             "Score of 60 should pass (>= 60). Change > to >=."),
    TestCase("H9", 'class_statistics({"A":85,"B":55})["passing"] == 1',
             lambda ns: _run('class_statistics({"A":85,"B":55})["passing"]', 1, ns),
             "55 < 60, only A passes."),
]


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
        task_id="easy", difficulty="easy",
        description="Fix 1 off-by-one bug in a statistics module",
        context=(
            "You are debugging a Python statistics module used by a data pipeline. "
            "The module has calculate_average and find_maximum. "
            "3 out of 5 tests are failing — something is wrong with calculate_average. "
            "Inspect the function, identify the bug, propose a fix."
        ),
        buggy_functions=dict(EASY_FUNCTIONS),
        tests=list(EASY_TESTS),
        max_operations=10, num_bugs=1,
    )


def get_medium_task() -> Task:
    return Task(
        task_id="medium", difficulty="medium",
        description="Fix 2 independent bugs in a text processing module",
        context=(
            "You are debugging a text utility library with reverse_words, truncate_text, "
            "and count_words. 4 out of 7 tests are failing — two different functions have "
            "independent bugs. Find and fix both."
        ),
        buggy_functions=dict(MEDIUM_FUNCTIONS),
        tests=list(MEDIUM_TESTS),
        max_operations=15, num_bugs=2,
    )


def get_hard_task() -> Task:
    return Task(
        task_id="hard", difficulty="hard",
        description="Fix 3 bugs in a grade calculator — two are interdependent",
        context=(
            "You are debugging a university grade system: letter_grade, weighted_average, "
            "calculate_final_grade, class_statistics. 6 of 9 tests fail. "
            "WARNING: Two of the three bugs are interdependent — fixing one without the "
            "other produces NO improvement. You must understand both bugs before fixing "
            "either. The third bug is independent."
        ),
        buggy_functions=dict(HARD_FUNCTIONS),
        tests=list(HARD_TESTS),
        max_operations=20, num_bugs=3,
    )


TASKS = {
    "easy": get_easy_task,
    "medium": get_medium_task,
    "hard": get_hard_task,
}
