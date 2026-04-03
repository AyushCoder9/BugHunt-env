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
