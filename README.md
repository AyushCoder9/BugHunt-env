---
title: BugHunt
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# 🔍 BugHunt — AI Debugging Arena

**BugHunt** is an OpenEnv-compliant reinforcement learning environment where AI agents learn to debug Python code through systematic investigation and targeted fixes.

> **Unique differentiator:** Hard mode features **interdependent bugs** — fixing one without the other yields *zero improvement*. Agents must reason about bug coupling, not just pattern-match.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   BugHunt v2.0                      │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  Gradio  │ Analytics│Curriculum│  Bug Dep │ Leader- │
│  Web UI  │ Tracking │ Learning │  Graphs  │  board  │
├──────────┴──────────┴──────────┴──────────┴─────────┤
│         OpenEnv Core SDK (Environment)              │
│    reset() · step() · state · create_app()          │
├─────────────────────────────────────────────────────┤
│          Sandboxed Python Execution                 │
│    Restricted builtins · No imports · No eval       │
└─────────────────────────────────────────────────────┘
```

## 🎯 Concept

Agents receive buggy Python functions and must:
1. **Inspect** function source code (free — no penalty)
2. **Run tests** to identify failures and read diagnostic hints (free)
3. **Propose fixes** — submit corrected function definitions (scored)
4. **Submit** to finalize their solution

This mirrors real-world debugging: gather information → form hypotheses → apply fixes → validate.

## 🧩 Tasks

| Task | Difficulty | Bugs | Tests | Max Ops | Key Challenge |
|------|-----------|------|-------|---------|---------------|
| `easy` | ⭐ | 1 | 5 | 10 | Off-by-one divisor in `calculate_average` |
| `medium` | ⭐⭐ | 2 | 7 | 15 | Two independent bugs in text processing |
| `hard` | ⭐⭐⭐ | 3 | 9 | 20 | **Two interdependent bugs** + one independent |

### Hard Mode — Interdependent Bugs 🔗

The hard task features a unique challenge: two of three bugs are **interdependent**. Bug 1 (wrong operator in `weighted_average`) **masks** Bug 2 (swapped arguments in `calculate_final_grade`). Fixing Bug 2 alone produces zero observable improvement because the arithmetic is still wrong. The agent must understand both bugs before fixing either — a test of genuine debugging reasoning, not just pattern matching.

```
H_BUG1 (weighted_average: + → *)  ──masks──▶  H_BUG2 (calculate_final_grade: args swapped)
                                                    │
H_BUG3 (class_statistics: > → >=)  ◀──independent──┘
```

## 🚀 Features

### Core RL Environment
- ✅ **OpenEnv SDK v2** — Full compliance with `openenv-core` types and server
- ✅ **Deterministic grading** — No randomness, no LLM-as-judge
- ✅ **Sandboxed execution** — Restricted builtins, no imports, no file access
- ✅ **Concurrent sessions** — Via OpenEnv WebSocket protocol

### Advanced Features
- 🎓 **Curriculum Learning** — Auto-promotes difficulty when avg score > 0.8
- 📊 **Analytics & Metrics** — Per-task episode stats, reward distributions
- 🏆 **Leaderboard** — Top 10 scores per difficulty level
- 🔗 **Bug Dependency Graphs** — Structured metadata for interdependent bugs
- 🎨 **Interactive Web UI** — Beautiful Gradio playground at `/web`
- 📋 **Capabilities API** — Machine-readable feature discovery

## 📋 Action Space

| Action | Parameters | Reward | Description |
|--------|-----------|--------|-------------|
| `inspect_function` | `function_name` | `0.0` | Read function source code |
| `run_test` | `test_id` | `0.0` | Execute test, see pass/fail + hint |
| `propose_fix` | `function_name`, `new_code` | `Δscore` or `-0.05` | Replace function with fix |
| `submit` | — | `final_score` | Finalize episode |

## 📊 API Endpoints

### Core (OpenEnv SDK)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take an action |
| `/state` | GET | Current state |
| `/ws` | WS | WebSocket sessions |
| `/docs` | GET | Swagger API docs |
| `/web` | GET | Interactive Gradio UI |

### Custom Extensions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analytics` | GET | Aggregate episode metrics |
| `/analytics/record` | POST | Record episode result |
| `/leaderboard` | GET | Top scores per task |
| `/curriculum` | GET | Curriculum learning status |
| `/curriculum/step` | POST | Record & check promotion |
| `/tasks/info` | GET | Task metadata & challenges |
| `/tasks/dependency_graph/{id}` | GET | Bug dependency graph |
| `/env/capabilities` | GET | Feature discovery |

## 🎮 Quick Start

### Using the Client SDK

```python
from client import BugHuntEnv
from models import BugHuntAction

async with BugHuntEnv(base_url="https://ayushxx9-bughunt-env.hf.space") as env:
    result = await env.reset(task_id="easy")
    result = await env.inspect_function("calculate_average")
    result = await env.run_test("E1")
    result = await env.propose_fix(
        "calculate_average",
        "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
    )
    result = await env.submit()
    print(f"Score: {result.reward}")
```

### Using HTTP

```bash
# Health check
curl https://ayushxx9-bughunt-env.hf.space/health

# Reset
curl -X POST https://ayushxx9-bughunt-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Step
curl -X POST https://ayushxx9-bughunt-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect_function", "function_name": "calculate_average"}}'

# Bug dependency graph (hard mode)
curl https://ayushxx9-bughunt-env.hf.space/tasks/dependency_graph/hard
```

### Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# With web UI enabled
ENABLE_WEB_INTERFACE=true uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## 🏆 Reward Design

| Event | Reward | Rationale |
|-------|--------|-----------|
| `inspect_function` | `0.0` | Free information gathering |
| `run_test` | `0.0` | Free information gathering |
| `propose_fix` (improves) | `Δscore` | Positive delta encourages progress |
| `propose_fix` (no improvement) | `-0.05` | Small penalty prevents guess-and-check |
| Invalid action | `-0.05` | Penalty for errors |
| `submit` | `final_score ∈ [0, 1]` | Fraction of tests passing |

## 🎓 Curriculum Learning

BugHunt supports automatic difficulty progression:

```
easy  ──(avg > 0.8)──▶  medium  ──(avg > 0.8)──▶  hard
```

Use `/curriculum` to check current level and `/curriculum/step` to record scores. The controller auto-promotes when the sliding window average exceeds the threshold.

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model for inference |
| `HF_TOKEN` | — | HuggingFace / API token |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |
| `ENABLE_WEB_INTERFACE` | `false` | Enable Gradio at `/web` |

## 📝 Technical Details

- **SDK:** `openenv-core` with `Environment`, `Action`, `Observation`, `State` types
- **Server:** `create_app()` factory with custom `gradio_builder`
- **Grading:** 100% deterministic — no randomness, no LLM-as-judge
- **Sandbox:** Code execution uses restricted builtins
- **Concurrency:** Supports concurrent sessions via WebSocket
- **Analytics:** In-memory episode tracking with leaderboard
- **Curriculum:** Sliding window auto-promotion across difficulty tiers

## 📄 License

MIT
