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

# 🔍 BugHunt — Python Debugging RL Environment

**BugHunt** is an OpenEnv-compliant reinforcement learning environment where AI agents learn to debug Python code through systematic investigation and targeted fixes.

## 🎯 Concept

Agents receive buggy Python functions and must:
1. **Inspect** function source code (free — no penalty)
2. **Run tests** to identify failures and read hints (free)
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

The hard task features a unique challenge: two of three bugs are **interdependent**. Fixing Bug 2 alone produces zero observable improvement because Bug 1 masks its effect. The agent must understand both bugs before fixing either — a test of genuine debugging reasoning, not just pattern matching.

## 📋 Action Space

| Action | Parameters | Reward | Description |
|--------|-----------|--------|-------------|
| `inspect_function` | `function_name` | `0.0` | Read function source code |
| `run_test` | `test_id` | `0.0` | Execute test, see pass/fail + hint |
| `propose_fix` | `function_name`, `new_code` | `Δscore` or `-0.05` | Replace function with fix |
| `submit` | — | `final_score` | Finalize episode |

### Action JSON Format

```json
{"action_type": "inspect_function", "function_name": "calculate_average"}
{"action_type": "run_test", "test_id": "E1"}
{"action_type": "propose_fix", "function_name": "calculate_average", "new_code": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"}
{"action_type": "submit"}
```

## 📊 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Episode complete? |
| `reward` | `float \| null` | Reward for last action |
| `task_id` | `str` | Current task ID |
| `task_context` | `str` | Natural language task description |
| `available_functions` | `list[str]` | Function names available to inspect |
| `available_tests` | `list[str]` | Test IDs available to run |
| `inspected_functions` | `dict[str, str]` | Source code inspected so far |
| `test_results` | `list[dict]` | Status + hints per test |
| `operations_log` | `list[str]` | History of actions taken |
| `operations_remaining` | `int` | Ops left before auto-submit |
| `current_score` | `float` | `tests_passed / tests_total` |
| `tests_passed` | `int` | Number of passing tests |
| `tests_total` | `int` | Total number of tests |
| `message` | `str` | Human-readable status message |

## 🏆 Reward Design

| Event | Reward | Rationale |
|-------|--------|-----------|
| `inspect_function` | `0.0` | Free information gathering |
| `run_test` | `0.0` | Free information gathering |
| `propose_fix` (improves score) | `new_score - old_score` | Positive delta |
| `propose_fix` (no improvement) | `-0.05` | Small penalty for bad fixes |
| Invalid action | `-0.05` | Penalty for errors |
| `submit` | `final_score ∈ [0, 1]` | Fraction of tests passing |

## 🚀 Quick Start

### Using the Client

```python
from client import BugHuntEnv
from models import BugHuntAction

# Async usage
async with BugHuntEnv(base_url="https://ayushxx9-bughunt-env.hf.space") as env:
    result = await env.reset(task_id="easy")
    result = await env.inspect_function("calculate_average")
    result = await env.run_test("E1")
    result = await env.propose_fix(
        "calculate_average",
        "def calculate_average(numbers):\\n    if not numbers:\\n        return 0\\n    return sum(numbers) / len(numbers)\\n"
    )
    result = await env.submit()
    print(f"Score: {result.reward}")
```

### Using HTTP Directly

```bash
# Health check
curl https://ayushxx9-bughunt-env.hf.space/health

# Reset
curl -X POST "https://ayushxx9-bughunt-env.hf.space/reset?task_id=easy"

# Step
curl -X POST https://ayushxx9-bughunt-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_function", "function_name": "calculate_average"}'
```

### Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## 📈 Expected Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| easy | ~1.000 | Off-by-one trivially found |
| medium | ~0.857 | Both text bugs usually found |
| hard | ~0.556 | Interdependency is challenging |
| **average** | **~0.804** | |

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model for inference |
| `HF_TOKEN` | — | HuggingFace / API token |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |

## 📝 Technical Details

- **SDK:** Built on `openenv-core` with proper `Environment`, `Action`, `Observation`, `State` types
- **Grading:** 100% deterministic — no randomness, no LLM-as-judge
- **Sandbox:** Code execution uses restricted builtins (no `import`, no `eval`, no file access)
- **Concurrency:** Supports concurrent sessions via OpenEnv WebSocket protocol

## 📄 License

MIT
