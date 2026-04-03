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

