# server/__init__.py
"""BugHunt environment server package."""

try:
    from .environment import BugHuntEnvironment
except ImportError:
    pass

__all__ = ["BugHuntEnvironment"]
