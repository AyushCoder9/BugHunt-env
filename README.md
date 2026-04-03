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

# BugHunt: Interactive RL Debugging Environment

BugHunt is an OpenEnv-compliant reinforcement learning environment developed for the Meta x Hugging Face OpenEnv Hackathon. It challenges AI agents to diagnose and repair software bugs within a controlled, observable sandbox.

## Overview

Unlike static code repair benchmarks, BugHunt implements a dynamic debugging trajectory. Agents are provided with a buggy Python module and must manage a finite operation budget to inspect source code, execute tests, and verify patches.

### Key Technical Challenges

The environment focuses on "Hard" debugging scenarios that defeat naive LLM strategies. Specifically, the **Hard Task** features an interdependency mechanic where two separate bugs must be identified and fixed simultaneously; patching only one results in zero reward, forcing the agent to demonstrate genuine causal reasoning.

## Task Specifications

| Difficulty | Module | Bugs | Tests | Max Operations |
|:---|:---|:---:|:---:|:---:|
| **Easy** | stats_module | 1 | 5 | 10 |
| **Medium** | text_processor | 2 | 7 | 15 |
| **Hard** | grade_calculator | 3 | 9 | 20 |

### 1. Easy Task
Focuses on a standard off-by-one error in a statistics utility. Ideal for verifying baseline operation logic.

### 2. Medium Task
Features two independent logic errors in string manipulation and text slicing routines. Requires sequential diagnostic steps.

### 3. Hard Task
Includes three bugs, two of which are procedurally interdependent. The agent must analyze cross-function calls within a grade calculation system to find a hidden arithmetic error that masks an argument-order bug.

## Environment Mechanics

### Action Space
Agents interact via a structured JSON API:
- `inspect_function`: Retrieves raw source code for a specific function (0 cost).
- `run_test`: Executes a test case and returns the traceback/hint (0 cost).
- `propose_fix`: Replaces a function implementation with user-provided code (1 cost).
- `submit`: Terminates the episode and returns the final score (1 cost).

### Observation Space
The state returned after every action includes:
- `inspected_functions`: A map of function names to their current source code.
- `test_results`: Detailed status and failure hints for every test in the suite.
- `current_score`: The current ratio of passing tests (0.0 to 1.0).
- `metadata`: Operations remaining, task context, and available function symbols.

### Reward Signal
Rewards are dense and deterministic. Every `propose_fix` that increases the `current_score` generates a positive reward signal (`new_score - prev_score`). Invalid actions or regressions incur a `-0.05` penalty.

## Implementation Details

- **Deterministic Graders**: All tasks use fixed, non-flaky testing logic.
- **Provider Fallback Agent**: The provided `inference.py` implementation uses an automated fallback loop (OpenAI -> Groq -> Gemini) to ensure reliability during rate-limiting or quota exhaustion.
- **Security**: Function execution is wrapped in a shared, restricted namespace with safe `__builtins__` to prevent exploit-driven solutions.

## Setup and Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Baseline Agent Execution
```bash
cp .env.example .env
# Configure your API keys in .env
python3 inference.py
```

## Performance Benchmarks

Results using the baseline `inference.py` agent (GPT-4o-mini / Llama-3.3-70B):

| Task | Score | Latency (Avg) |
|:---|:---:|:---:|
| Easy | 1.000 | < 10s |
| Medium | 1.000 | < 30s |
| Hard | 1.000 | < 60s |
| **Average** | **1.000** | **~90s Total** |

## API Reference

The environment exposes several standard endpoints:
- `GET /health`: System liveness check.
- `POST /reset`: Initialize a task episode.
- `POST /step`: Execute an agent action.
- `GET /state`: View non-destructive environment metadata.
- `GET /web`: Interactive diagnostic dashboard.
- `GET /docs`: Full OpenAPI/Swagger specification.
