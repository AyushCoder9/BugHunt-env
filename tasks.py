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
