#!/usr/bin/env python3
"""
BugHunt — Baseline Inference Script

Required by hackathon: must be named inference.py, in root directory.
Uses OpenAI-compatible client (required by competition rules).
Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars.

Structured stdout logging format:
    [BUGHUNT] key=value key2=value2 ...
"""
import json
import os
import sys
import time

import requests
from openai import OpenAI

# ── Configuration from environment variables ────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 18
TASKS = ["easy", "medium", "hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

SYSTEM_PROMPT = """You are an expert Python debugger in the BugHunt RL environment.

ACTIONS (reply with ONLY valid JSON):
{"action_type": "inspect_function", "function_name": "<name>"}
{"action_type": "run_test", "test_id": "<id>"}
{"action_type": "propose_fix", "function_name": "<name>", "new_code": "<full def block>"}
{"action_type": "submit"}

STRATEGY:
1. inspect_function for each available function
2. run_test for each test to see which fail and read hints
3. propose_fix for each buggy function (new_code must start with "def ")
4. submit when all tests show pass or score=1.0

RULES:
- new_code must start with "def " — complete function definition only
- No imports, no eval(), no exec(), no os/sys in new_code
- Reply JSON only — no explanation, no markdown backticks
"""


def log(msg: str, **kwargs):
    """Structured stdout logging."""
    parts = [f"[BUGHUNT] {msg}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def reset_env(task_id: str) -> dict:
    """Reset the environment for a specific task."""
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        headers={"x-session-id": "123"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()





def step_env(action: dict) -> dict:
    """Take a step in the environment."""
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, headers={"x-session-id": "123"}, timeout=15)
    r.raise_for_status()
    return r.json()


def build_prompt(obs: dict, step: int) -> str:
    """Build the user prompt from the current observation."""
    lines = [
        f"=== BUGHUNT {obs.get('task_id', '').upper()} | Step {step} ===",
        f"Score: {obs.get('current_score', 0):.2f} "
        f"({obs.get('tests_passed', 0)}/{obs.get('tests_total', 0)} tests) "
        f"| Ops left: {obs.get('operations_remaining', 0)}",
        f"Context: {obs.get('task_context', '')}",
        "",
        "Functions: " + ", ".join(obs.get("available_functions", [])),
        "Tests: " + ", ".join(obs.get("available_tests", [])),
        "",
    ]

    inspected = obs.get("inspected_functions", {})
    if inspected:
        lines.append("=== SOURCE CODE ===")
        for name, src in inspected.items():
            lines.append(f"[{name}]\n{src}")

    lines.append("=== TEST RESULTS ===")
    for tr in obs.get("test_results", []):
        icon = {
            "pass": "✅",
            "fail": "❌",
            "error": "💥",
            "not_run": "⬜",
        }.get(tr.get("status", "not_run"), "⬜")
        lines.append(f"{icon} [{tr['test_id']}] {tr['description']}")
        if tr.get("output"):
            lines.append(f"   Hint: {tr['output']}")

    lines.append("=== OPS LOG ===")
    for op in obs.get("operations_log", [])[-6:]:
        lines.append(f"  {op}")

    lines.append(
        f"\nLast: {obs.get('message', '')}\n\nYour next action (JSON only):"
    )
    return "\n".join(lines)


def parse_action(text: str) -> dict:
    """Parse LLM response text into an action dict."""
    text = text.strip()
    for fence in ["```json", "```"]:
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
            break
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return {"action_type": "submit"}


def run_task(task_id: str) -> float:
    """Run a single task and return the final score."""
    print(f"[START] task={task_id}", flush=True)
    log("task_start", task_id=task_id)
    result = reset_env(task_id)
    obs = result["observation"]
    done = result.get("done", False)
    log(
        "task_reset",
        task_id=task_id,
        description=obs.get("task_description", ""),
        tests_total=obs.get("tests_total", 0),
    )

    final_score = 0.0
    for step in range(1, MAX_STEPS + 1):
        if done:
            final_score = result.get("reward") or 0.0
            print(f"[END] task={task_id} score={final_score:.3f} steps={step - 1}", flush=True)
            log("task_done", task_id=task_id, step=step - 1, score=f"{final_score:.3f}")
            break

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(obs, step)},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            log("llm_error", error=str(exc))
            response_text = '{"action_type":"submit"}'

        action = parse_action(response_text)
        atype = action.get("action_type", "?")
        detail = action.get("function_name") or action.get("test_id", "")
        log(
            "step",
            task_id=task_id,
            step=step,
            action=atype,
            target=detail,
        )

        result = step_env(action)
        obs = result.get("observation", {})
        done = result.get("done", False)
        print(f"[STEP] step={step} reward={result.get('reward', 0.0):.3f}", flush=True)
        log(
            "step_result",
            task_id=task_id,
            step=step,
            score=f"{obs.get('current_score', 0):.2f}",
            reward=f"{result.get('reward', 0):+.3f}",
        )
    else:
        # Force submit if max steps reached
        result = step_env({"action_type": "submit"})
        final_score = result.get("reward") or 0.0
        print(f"[END] task={task_id} score={final_score:.3f} steps={MAX_STEPS}", flush=True)
        log("forced_submit", task_id=task_id, score=f"{final_score:.3f}")

    return final_score


def main():
    """Run the baseline inference across all tasks."""
    log("inference_start", model=MODEL_NAME, env=ENV_BASE_URL)

    # Health check
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        log("health_ok", response=str(r.json()))
    except Exception as exc:
        log("health_fail", error=str(exc))
        print(
            "\n❌ Cannot reach environment. Start server:\n"
            "  uvicorn server.app:app --host 0.0.0.0 --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)

    scores = {}
    t0 = time.time()

    for task_id in TASKS:
        scores[task_id] = run_task(task_id)

    avg = sum(scores.values()) / len(scores)
    elapsed = time.time() - t0

    # Final summary
    log("inference_complete", elapsed_s=f"{elapsed:.1f}")
    print(f"\n{'=' * 55}")
    print(f"  BASELINE SCORES")
    print(f"{'=' * 55}")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 30)
        print(f"  {tid:8s}  {sc:.3f}  {bar}")
    print(f"  {'average':8s}  {avg:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 55}")

    return scores


if __name__ == "__main__":
    main()
