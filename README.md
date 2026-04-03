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

# 🔍 BugHunt

An **OpenEnv** RL environment where AI agents debug Python code.

The agent is given a module with planted bugs. It can **inspect** function source code, **run** individual test cases to see what's failing, and **propose fixes** by submitting corrected function implementations. Every test that starts passing earns a reward signal.

## Why this is hard for LLMs

Standard LLM benchmarks give the model the full codebase to read. BugHunt forces the agent to **choose** what to inspect — it has a limited operation budget and must reason about which functions are buggy before spending operations on fixes. The **hard task** has two interdependent bugs: fixing one without understanding the other produces zero improvement, forcing genuine causal reasoning.

---

## Tasks

### Easy — `stats_module` (1 bug, 5 tests, max 10 ops)
**Bug:** `calculate_average` divides by `len(numbers) - 1` instead of `len(numbers)`. Causes `ZeroDivisionError` on single-element lists and wrong values elsewhere.

**Optimal:** inspect → run a test → propose fix → submit (4 ops)

### Medium — `text_processor` (2 bugs, 7 tests, max 15 ops)
**Bug 1:** `reverse_words` returns words in original order (missing `[::-1]`).  
**Bug 2:** `truncate_text` slices `text[:max_length]` instead of `text[:max_length-3]`, so the `"..."` suffix makes the result exceed `max_length`.

**Optimal:** inspect both → fix both → submit (6 ops)

### Hard — `grade_calculator` (3 bugs, 9 tests, max 20 ops)
**Bug 1** (`weighted_average`): uses `score + weight` instead of `score * weight`.  
**Bug 2** (`calculate_final_grade`): `midterm` and `final_exam` are **swapped** in the call to `weighted_average`. This bug is completely **masked by Bug 1** — fixing Bug 2 alone changes nothing observable. The agent must diagnose both before either fix helps.  
**Bug 3** (`class_statistics`): uses `> 60` instead of `>= 60` for the passing threshold — independent.

**Interdependency:** Tests H4 and H5 fail due to both Bug 1 and Bug 2 together. Fixing only Bug 1 leaves H4/H5 failing (wrong argument order). Fixing only Bug 2 leaves H4/H5 failing (still wrong arithmetic). Both must be fixed.

---

## Action Space

| `action_type` | Extra fields | Cost | Description |
|---|---|---|---|
| `inspect_function` | `function_name` | 0.0 | Read source code |
| `run_test` | `test_id` | 0.0 | Run one test, get pass/fail + hint |
| `propose_fix` | `function_name`, `new_code` | varies | Replace function implementation |
| `submit` | — | — | Finalise episode |

`new_code` must be a complete `def` block. No imports, no `os`/`sys`/`eval`.

## Observation Space

| Field | Type | Description |
|---|---|---|
| `done` | bool | Episode finished |
| `reward` | float \| null | Step reward |
| `task_context` | str | Narrative about the buggy module |
| `available_functions` | list[str] | Functions that can be inspected/fixed |
| `available_tests` | list[str] | Test IDs that can be run |
| `inspected_functions` | dict | Source code of functions the agent has read |
| `test_results` | list | Status + hint per test (pass/fail/error/not_run) |
| `operations_log` | list | History of actions taken |
| `operations_remaining` | int | Budget left |
| `current_score` | float | tests_passed / tests_total |
| `message` | str | Human-readable feedback |

## Reward Function

| Situation | Reward |
|---|---|
| `inspect_function` or `run_test` | `0.0` (free) |
| `propose_fix` that increases score | `new_score − prev_score` (positive) |
| `propose_fix` with no improvement | `−0.05` |
| Invalid action | `−0.05` |
| `submit()` or auto-submit | `final_score ∈ [0, 1]` |

The reward is never sparse — every successful fix yields a positive signal immediately.

---

## Setup

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker
```bash
docker build -t bughunt .
docker run -p 7860:7860 bughunt
```

### Quick test
```bash
# Health
curl http://localhost:7860/health

# Reset easy task
curl -X POST "http://localhost:7860/reset?task_id=easy"

# Inspect a function
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_function", "function_name": "calculate_average"}'

# Propose a fix
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "propose_fix",
    "function_name": "calculate_average",
    "new_code": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"
  }'

# Submit
curl -X POST http://localhost:7860/step -d '{"action_type": "submit"}'
```

### Baseline inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

---

## Baseline Scores

Measured with `gpt-4o-mini` (temperature 0.1):

| Task | Score | Notes |
|---|---|---|
| easy | 1.000 | Finds and fixes the off-by-one immediately |
| medium | 0.857 | Fixes reverse_words correctly; truncate_text partially |
| hard | 0.556 | Fixes Bug 3 independently; struggles with interdependency |
| **average** | **0.804** | |

The hard task genuinely challenges frontier models because the standard strategy (fix one bug at a time and check if score improves) fails for the interdependent pair.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness |
| POST | `/reset?task_id=easy` | Start episode |
| POST | `/step` | Take action |
| GET | `/state` | Episode metadata |
| WS | `/ws` | Persistent WebSocket session |
| GET | `/docs` | OpenAPI docs |
| GET | `/web` | Web UI |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | — | API key |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |
