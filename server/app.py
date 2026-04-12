# server/app.py
"""
FastAPI application entry point for the BugHunt environment.

Uses openenv.core.env_server.http_server.create_app to create all standard
OpenEnv endpoints: /health, /reset, /step, /state, /ws, /docs, /web
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
except ImportError:
    from models import BugHuntAction, BugHuntObservation
    from server.environment import BugHuntEnvironment


def create_bughunt_environment() -> BugHuntEnvironment:
    """Factory function for BugHuntEnvironment instances."""
    return BugHuntEnvironment()


# create_app takes a factory function, not a class
app = create_app(
    create_bughunt_environment,
    BugHuntAction,
    BugHuntObservation,
    env_name="bughunt_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the BugHunt environment server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()