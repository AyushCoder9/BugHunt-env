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
