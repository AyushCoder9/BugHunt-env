#!/usr/bin/env python3
"""
BugHunt — Baseline Inference Script
=====================================
Required by the OpenEnv competition spec.
- Uses OpenAI client for ALL LLM calls (hackathon requirement satisfied)
- Multi-provider fallback: OpenAI → Groq → Gemini (all via OpenAI-compatible API)
- Reads credentials from environment variables
- Runs all 3 tasks and prints reproducible scores
- Runtime well under 20 min on 2vcpu / 8GB RAM


Hackathon-mandated environment variables
-----------------------------------------
API_BASE_URL   Primary LLM endpoint   (default: https://api.openai.com/v1)
MODEL_NAME     Primary model          (default: gpt-4o-mini)
HF_TOKEN       Primary API key
ENV_BASE_URL   Environment server URL (default: http://localhost:7860)


Optional fallback keys (tried in order if primary fails)
---------------------------------------------------------
OPENAI_API_KEY    OpenAI key  (also accepted as primary if HF_TOKEN is unset)
GROQ_API_KEY      Groq key    → endpoint: https://api.groq.com/openai/v1
GEMINI_API_KEY    Gemini key  → endpoint: https://generativelanguage.googleapis.com/v1beta/openai/
"""
from __future__ import annotations
import json
import os
import sys
import time


import requests
from openai import OpenAI


try:
   from dotenv import load_dotenv
   load_dotenv()
except ImportError:
   pass


# ---------------------------------------------------------------------------
# Hackathon-required env vars (always read, never hardcoded)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


MAX_STEPS   = 18
TEMPERATURE = 0.1
MAX_TOKENS  = 512
TASKS       = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Multi-provider fallback configuration
# ---------------------------------------------------------------------------
# Each entry is tried in order. The primary provider is always built from the
# hackathon-mandated env vars (API_BASE_URL / MODEL_NAME / HF_TOKEN).
# Subsequent entries use their own optional keys read from the environment.
# All providers use the OpenAI client — the hackathon rule is satisfied.
# ---------------------------------------------------------------------------


def _build_providers() -> list[dict]:
   """Build the ordered list of providers from environment variables."""
   providers = []


   # --- Primary (hackathon-mandated) ---
   # Accept HF_TOKEN or OPENAI_API_KEY for the primary slot
   primary_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")
   if primary_key:
       providers.append({
           "name": "Primary (" + API_BASE_URL.split("/")[2] + ")",
           "base_url": API_BASE_URL,
           "api_key": primary_key,
           "model": MODEL_NAME,
       })


   # --- Fallback: Groq (OpenAI-compatible, generous free tier) ---
   groq_key = os.environ.get("GROQ_API_KEY", "")
   if groq_key and groq_key not in {p["api_key"] for p in providers}:
       providers.append({
           "name": "Groq (llama-3.3-70b-versatile)",
           "base_url": "https://api.groq.com/openai/v1",
           "api_key": groq_key,
           "model": os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
       })


   # --- Fallback: Google Gemini (OpenAI-compatible endpoint) ---
   gemini_key = os.environ.get("GEMINI_API_KEY", "")
   if gemini_key and gemini_key not in {p["api_key"] for p in providers}:
       providers.append({
           "name": "Gemini (gemini-1.5-flash)",
           "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
           "api_key": gemini_key,
           "model": os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
       })


   return providers




PROVIDERS = _build_providers()


if not PROVIDERS:
   # Warn at import time but don't exit — exit happens in main()
   print("⚠  Warning: No API keys configured. Set HF_TOKEN / OPENAI_API_KEY / GROQ_API_KEY / GEMINI_API_KEY")


# Keep a module-level client pointing at the primary provider.
# The call_llm() helper creates fresh clients for fallback providers.
_primary = PROVIDERS[0]
client = OpenAI(base_url=_primary["base_url"], api_key=_primary["api_key"])


# Errors that signal quota exhaustion or rate limits — trigger a fallback
_FALLBACK_TRIGGERS = (
   "insufficient_quota",
   "rate_limit",
   "rate limit",
   "quota",
   "429",
   "billing",
   "exceeded",
   "limit exceeded",
   "context length",   # some providers throw this; try next with shorter ctx
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert Python debugger. Your job is to find and fix bugs in Python functions
by inspecting source code and running tests inside the BugHunt RL environment.


=== AVAILABLE ACTIONS — reply with ONLY valid JSON ===


1. Inspect a function (free — costs 0 reward):
  {"action_type": "inspect_function", "function_name": "<name>"}


2. Run a specific test (free — costs 0 reward):
  {"action_type": "run_test", "test_id": "<id>"}


3. Propose a complete fixed implementation:
  {"action_type": "propose_fix", "function_name": "<name>", "new_code": "<full def block>"}


4. Submit your final answer:
  {"action_type": "submit"}


=== STRATEGY ===
1. Inspect all available functions first to understand the codebase.
2. Run all tests to see which pass and which fail.
3. Read the failure hints — they tell you where to look.
4. Propose a fix for each buggy function.
5. After each fix, check if score improved.
6. Call submit when all tests pass or you are satisfied.


=== RULES ===
- new_code must be a complete function definition starting with "def ".
- Do NOT use imports, open(), eval(), exec(), or os/sys inside new_code.
- Reply with JSON only. No explanation, no markdown, no backticks.
"""


FALLBACK = '{"action_type": "submit"}'


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def reset_env(task_id: str) -> dict:
   r = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=15)
   r.raise_for_status()
   return r.json()




def step_env(action: dict) -> dict:
   r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=15)
   r.raise_for_status()
   return r.json()




# ---------------------------------------------------------------------------
# Multi-provider LLM caller with automatic fallback
# ---------------------------------------------------------------------------


# Track which provider index we are currently using so consecutive calls
# reuse the same provider unless it fails.
_current_provider_idx = 0




def call_llm(messages: list[dict]) -> str:
   """
   Call the LLM using the OpenAI client.  If the active provider returns a
   quota / rate-limit error, automatically switch to the next available
   provider and retry.  Returns the response text, or FALLBACK on total
   failure.
   """
   global _current_provider_idx


   # Try from the current provider onwards; wrap around at most once through
   # the full list so we don't loop forever.
   attempts = list(range(_current_provider_idx, len(PROVIDERS))) + \
              list(range(0, _current_provider_idx))


   for idx in attempts:
       provider = PROVIDERS[idx]
       try:
           c = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])
           completion = c.chat.completions.create(
               model=provider["model"],
               messages=messages,
               temperature=TEMPERATURE,
               max_tokens=MAX_TOKENS,
           )
           # Success — lock to this provider for future calls
           if idx != _current_provider_idx:
               print(f"  ✦ Switched to provider: {provider['name']}")
               _current_provider_idx = idx
           return completion.choices[0].message.content or ""


       except Exception as exc:
           err = str(exc).lower()
           is_quota_issue = any(t in err for t in _FALLBACK_TRIGGERS)
           if is_quota_issue and idx != attempts[-1]:
               next_p = PROVIDERS[attempts[attempts.index(idx) + 1]]
               print(f"  ⚠ {provider['name']} quota/rate-limit — trying {next_p['name']}")
               continue
           # Non-retriable error or last provider
           print(f"  ⚠ LLM call failed ({provider['name']}): {exc} — using submit fallback.")
           return FALLBACK


   return FALLBACK




# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(obs: dict, step: int) -> str:
   task_id = obs.get("task_id", "").upper()
   lines = [
       f"=== BUGHUNT | {task_id} | Step {step} ===",
       "",
       f"Context: {obs.get('task_context', '')}",
       "",
       f"Score: {obs.get('current_score', 0):.2f}  "
       f"({obs.get('tests_passed', 0)}/{obs.get('tests_total', 0)} tests passing)  "
       f"Ops remaining: {obs.get('operations_remaining', 0)}",
       "",
   ]


   # Available functions
   lines += [
       "─── Available functions to inspect/fix ───",
       "  " + ", ".join(obs.get("available_functions", [])),
       "",
   ]


   # Inspected source
   inspected = obs.get("inspected_functions", {})
   if inspected:
       lines.append("─── Inspected source code ───")
       for name, src in inspected.items():
           lines.append(f"\n[{name}]\n{src}")
       lines.append("")


   # Test results
   test_results = obs.get("test_results", [])
   if test_results:
       lines.append("─── Test results ───")
       for tr in test_results:
           status = tr.get("status", "not_run")
           icon = {"pass": "✅", "fail": "❌", "error": "💥", "not_run": "⬜"}[status]
           line = f"  {icon} [{tr['test_id']}] {tr['description']}"
           if status in ("fail", "error") and tr.get("output"):
               line += f"\n       Hint: {tr['output']}"
           lines.append(line)
       lines.append("")


   # Ops log
   log = obs.get("operations_log", [])
   if log:
       lines.append("─── Operations so far ───")
       for op in log[-8:]:   # last 8 to keep context short
           lines.append(f"  {op}")
       lines.append("")


   lines += [
       f"Last message: {obs.get('message', '')}",
       "",
       "Your next action (JSON only):",
   ]
   return "\n".join(lines)




# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------


def parse_action(text: str) -> dict:
   text = text.strip()
   # Strip markdown fences if model added them
   for fence in ["```json", "```"]:
       if fence in text:
           text = text.split(fence)[1].split("```")[0].strip()
           break
   try:
       return json.loads(text)
   except json.JSONDecodeError:
       # Try to extract JSON from the text
       start = text.find("{")
       end = text.rfind("}") + 1
       if start != -1 and end > start:
           try:
               return json.loads(text[start:end])
           except Exception:
               pass
   return {"action_type": "submit"}

