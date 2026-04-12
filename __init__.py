# __init__.py
"""
BugHunt — OpenEnv RL environment for debugging Python code.

Agents inspect functions, run tests, propose fixes, and submit solutions.
Three difficulty levels with deterministic grading.
"""

try:
    from .client import BugHuntEnv
    from .models import BugHuntAction, BugHuntObservation, BugHuntState
except ImportError:
    pass

__all__ = ["BugHuntEnv", "BugHuntAction", "BugHuntObservation", "BugHuntState"]
