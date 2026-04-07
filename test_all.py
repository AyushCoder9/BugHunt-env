#!/usr/bin/env python3
"""
BugHunt — Comprehensive verification test suite.
Tests every requirement from the hackathon spec.
"""
from __future__ import annotations
import json
import sys
import traceback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
   global PASS, FAIL
   if condition:
       PASS += 1
       print(f"  ✅ {name}")
   else:
       FAIL += 1
       msg = f"  ❌ {name}"
       if detail:
           msg += f"  — {detail}"
       print(msg)


def approx(a, b, tol=0.02):
   return abs(a - b) < tol




# ===================================================================
# SECTION 1: Unit tests — Environment logic
# ===================================================================
print("\n" + "=" * 65)
print("  SECTION 1: Environment unit tests")
print("=" * 65)


from environment import BugHuntEnvironment
from models import BugHuntAction


# --- 1a. EASY task — complete correct solve ---
print("\n--- 1a. EASY task — complete correct solve ---")
env = BugHuntEnvironment()
obs = env.reset("easy")
check("easy: done=False on reset", obs.done == False)
check("easy: initial score ~0.40", approx(obs.current_score, 0.40), f"got {obs.current_score:.3f}")
check("easy: tests_total=5", obs.tests_total == 5, f"got {obs.tests_total}")
check("easy: ops_remaining=10", obs.operations_remaining == 10, f"got {obs.operations_remaining}")
check("easy: calculate_average in available_functions", "calculate_average" in obs.available_functions)
check("easy: find_maximum in available_functions", "find_maximum" in obs.available_functions)


# Inspect is free
obs = env.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
check("easy: inspect reward=0.0", obs.reward == 0.0, f"got {obs.reward}")
check("easy: inspected_functions has calculate_average", "calculate_average" in obs.inspected_functions)
check("easy: source contains 'def calculate_average'", "def calculate_average" in obs.inspected_functions.get("calculate_average", ""))


# Run test is free
obs = env.step(BugHuntAction(action_type="run_test", test_id="E1"))
check("easy: run_test reward=0.0", obs.reward == 0.0, f"got {obs.reward}")
e1 = next((r for r in obs.test_results if r.test_id == "E1"), None)
check("easy: E1 status is fail", e1 is not None and e1.status == "fail", f"got {e1.status if e1 else 'None'}")
hint_ok = e1 and ("divisor" in e1.output.lower() or "off" in e1.output.lower() or "check" in e1.output.lower())
check("easy: E1 failure hint mentionable", hint_ok, f"hint: {e1.output if e1 else ''}")


# Correct fix
fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=fix))
check("easy: fix reward ~0.60", approx(obs.reward, 0.60), f"got {obs.reward:.3f}")
check("easy: score after fix ~1.00", approx(obs.current_score, 1.00), f"got {obs.current_score:.3f}")


# Submit
obs = env.step(BugHuntAction(action_type="submit"))
check("easy: done=True after submit", obs.done == True)
check("easy: submit reward ~1.00", approx(obs.reward, 1.00), f"got {obs.reward:.3f}")




# --- 1b. MEDIUM task — complete correct solve ---
print("\n--- 1b. MEDIUM task — complete correct solve ---")
obs = env.reset("medium")
check("medium: tests_total=7", obs.tests_total == 7, f"got {obs.tests_total}")
check("medium: ops_remaining=15", obs.operations_remaining == 15, f"got {obs.operations_remaining}")


fix_reverse = "def reverse_words(sentence):\n    if not sentence.strip():\n        return ''\n    return ' '.join(sentence.split()[::-1])\n"
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="reverse_words", new_code=fix_reverse))
check("medium: reverse_words fix reward>0", obs.reward > 0, f"got {obs.reward:.3f}")
check("medium: score after reverse fix ~0.71", approx(obs.current_score, 0.71), f"got {obs.current_score:.3f}")


fix_truncate = "def truncate_text(text, max_length):\n    if len(text) <= max_length:\n        return text\n    return text[:max_length - 3] + '...'\n"
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="truncate_text", new_code=fix_truncate))
check("medium: truncate fix reward>0", obs.reward > 0, f"got {obs.reward:.3f}")
check("medium: score after both ~1.00", approx(obs.current_score, 1.00), f"got {obs.current_score:.3f}")


obs = env.step(BugHuntAction(action_type="submit"))
check("medium: done=True", obs.done == True)
check("medium: submit reward ~1.00", approx(obs.reward, 1.00), f"got {obs.reward:.3f}")




# --- 1c. HARD task — interdependency ---
print("\n--- 1c. HARD task — interdependency ---")
obs = env.reset("hard")
check("hard: tests_total=9", obs.tests_total == 9, f"got {obs.tests_total}")
check("hard: ops_remaining=20", obs.operations_remaining == 20, f"got {obs.operations_remaining}")


# Bug 3 is independent
fix_stats = """def class_statistics(student_scores):
   if not student_scores:
       return {"mean": 0, "highest": 0, "lowest": 0, "passing": 0}
   scores = list(student_scores.values())
   passing = sum(1 for s in scores if s >= 60)
   return {
       "mean": round(sum(scores) / len(scores), 2),
       "highest": max(scores),
       "lowest": min(scores),
       "passing": passing,
   }
"""
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="class_statistics", new_code=fix_stats))
check("hard: bug3 fix reward>0 (independent)", obs.reward > 0, f"got {obs.reward:.3f}")


# Bug 2 ALONE must NOT improve grade tests
fix_order_only = """def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3, 0.3, 0.4)):
   if not assignments:
       return "F"
   avg_assignment = sum(assignments) / len(assignments)
   composite = weighted_average([avg_assignment, midterm, final_exam], list(weights))
   return letter_grade(composite)
"""
score_before_bug2 = obs.current_score
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="calculate_final_grade", new_code=fix_order_only))
check("hard: bug2-alone reward<=0 (interdependency)", obs.reward <= 0, f"got {obs.reward:.3f}")
check("hard: score unchanged after bug2-alone", obs.current_score == score_before_bug2,
     f"was {score_before_bug2:.3f}, now {obs.current_score:.3f}")


# Bug 1 fix — now both bugs fixed
fix_weighted = """def weighted_average(scores, weights):
   if len(scores) != len(weights):
       return 0
   total = 0
   for score, weight in zip(scores, weights):
       total += score * weight
   return total
"""
obs = env.step(BugHuntAction(action_type="propose_fix", function_name="weighted_average", new_code=fix_weighted))
check("hard: bug1 fix reward>0 (both bugs now fixed)", obs.reward > 0, f"got {obs.reward:.3f}")


obs = env.step(BugHuntAction(action_type="submit"))
check("hard: done=True", obs.done == True)
check("hard: submit reward ~1.00", approx(obs.reward, 1.00), f"got {obs.reward:.3f}")




# --- 1d. Penalty and edge case tests ---
print("\n--- 1d. Penalty and edge case tests ---")
env2 = BugHuntEnvironment()
env2.reset("easy")


obs = env2.step(BugHuntAction(action_type="inspect_function", function_name="nonexistent_fn"))
check("penalty: bad function name reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")


obs = env2.step(BugHuntAction(action_type="run_test", test_id="ZZZ"))
check("penalty: bad test_id reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")


bad_fix = "def calculate_average(numbers):\n    import os\n    return 0\n"
obs = env2.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=bad_fix))
check("penalty: forbidden code reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")


syntax_err = "def calculate_average(numbers):\n    return sum(numbers) /\n"
obs = env2.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=syntax_err))
check("penalty: syntax error reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")


obs = env2.step(BugHuntAction(action_type="fly_to_the_moon"))
check("penalty: unknown action reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")


obs = env2.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
check("penalty: inspect always free reward=0.0", obs.reward == 0.0, f"got {obs.reward}")


obs = env2.step(BugHuntAction(action_type="run_test", test_id="E1"))
check("penalty: run_test always free reward=0.0", obs.reward == 0.0, f"got {obs.reward}")


# After submit
obs = env2.step(BugHuntAction(action_type="submit"))
obs2 = env2.step(BugHuntAction(action_type="inspect_function", function_name="calculate_average"))
check("penalty: after submit done=True", obs2.done == True)


# No improvement fix
env3 = BugHuntEnvironment()
env3.reset("easy")
wrong_fix = "def calculate_average(numbers):\n    return 999\n"
obs = env3.step(BugHuntAction(action_type="propose_fix", function_name="calculate_average", new_code=wrong_fix))
check("penalty: no-improvement fix reward=-0.05", obs.reward == -0.05, f"got {obs.reward}")




# --- 1e. Score range validation ---
print("\n--- 1e. Score range validation ---")
for tid in ["easy", "medium", "hard"]:
   env_v = BugHuntEnvironment()
   env_v.reset(tid)
   score = env_v._score()
   check(f"score_range: {tid} initial in [0,1]", 0.0 <= score <= 1.0, f"got {score}")
   obs = env_v.step(BugHuntAction(action_type="submit"))
   check(f"score_range: {tid} submit reward in [0,1]", 0.0 <= obs.reward <= 1.0, f"got {obs.reward}")
   check(f"score_range: {tid} submit done=True", obs.done == True)




# --- 1f. Determinism test ---
print("\n--- 1f. Determinism test ---")
def run_sequence(task_id):
   e = BugHuntEnvironment()
   e.reset(task_id)
   o = e.step(BugHuntAction(action_type="submit"))
   return o.reward


for tid in ["easy", "medium", "hard"]:
   scores = [run_sequence(tid) for _ in range(3)]
   check(f"determinism: {tid} same score x3", scores[0] == scores[1] == scores[2],
         f"got {scores}")




# ===================================================================
# SECTION 2: HTTP endpoint tests
# ===================================================================
print("\n" + "=" * 65)
print("  SECTION 2: HTTP endpoint tests")
print("=" * 65)


import requests
BASE = "http://localhost:7860"


# Health
r = requests.get(f"{BASE}/health")
check("http: /health status=200", r.status_code == 200)
check("http: /health body.status=healthy", r.json()["status"] == "healthy")


# Reset
r = requests.post(f"{BASE}/reset", params={"task_id": "easy"})
check("http: /reset status=200", r.status_code == 200)
data = r.json()
check("http: /reset has 'observation'", "observation" in data)
check("http: /reset has 'reward'", "reward" in data)
check("http: /reset has 'done'", "done" in data)
check("http: /reset done=False", data["done"] == False)
obs_d = data["observation"]
check("http: /reset obs has task_id", "task_id" in obs_d)
check("http: /reset obs task_id=easy", obs_d["task_id"] == "easy")
check("http: /reset obs has available_functions", "available_functions" in obs_d)
check("http: /reset obs has available_tests", "available_tests" in obs_d)
check("http: /reset obs has tests_total", "tests_total" in obs_d)


# Step — inspect
r = requests.post(f"{BASE}/step", json={"action_type": "inspect_function", "function_name": "calculate_average"})
check("http: inspect step status=200", r.status_code == 200)
check("http: inspect step reward=0.0", r.json()["reward"] == 0.0)


# Step — run_test
r = requests.post(f"{BASE}/step", json={"action_type": "run_test", "test_id": "E1"})
check("http: run_test step status=200", r.status_code == 200)


# Step — propose_fix
fix_code = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
r = requests.post(f"{BASE}/step", json={"action_type": "propose_fix", "function_name": "calculate_average", "new_code": fix_code})
check("http: propose_fix status=200", r.status_code == 200)
check("http: propose_fix reward>0", r.json()["reward"] > 0, f"got {r.json()['reward']}")


# Step — submit
r = requests.post(f"{BASE}/step", json={"action_type": "submit"})
check("http: submit status=200", r.status_code == 200)
check("http: submit done=True", r.json()["done"] == True)
check("http: submit reward=1.0", r.json()["reward"] == 1.0, f"got {r.json()['reward']}")


# State
r = requests.get(f"{BASE}/state")
check("http: /state status=200", r.status_code == 200)
state = r.json()
check("http: /state has episode_id", "episode_id" in state)
check("http: /state has step_count", "step_count" in state)
check("http: /state has task_id", "task_id" in state)


# All 3 tasks accessible
for tid in ["easy", "medium", "hard"]:
   r = requests.post(f"{BASE}/reset", params={"task_id": tid})
   check(f"http: reset {tid} ok", r.status_code == 200)
   check(f"http: reset {tid} task_id correct", r.json()["observation"]["task_id"] == tid)


# Docs endpoint
r = requests.get(f"{BASE}/docs")
check("http: /docs status=200", r.status_code == 200)


# Web UI
r = requests.get(f"{BASE}/web")
check("http: /web status=200", r.status_code == 200)




# ===================================================================
# SECTION 3: File structure checks
# ===================================================================
print("\n" + "=" * 65)
print("  SECTION 3: File structure & content checks")
print("=" * 65)


import os
ROOT = os.path.dirname(os.path.abspath(__file__))


required_files = [
   "inference.py", "server/app.py", "environment.py", "models.py", "tasks.py",
   "run_server.py", "Dockerfile", "requirements.txt", "openenv.yaml",
   "pyproject.toml", "README.md",
]
for f in required_files:
   check(f"file exists: {f}", os.path.isfile(os.path.join(ROOT, f)))


# Check inference.py uses OpenAI client
with open(os.path.join(ROOT, "inference.py")) as f:
   inf_src = f.read()
check("inference.py: imports OpenAI", "from openai import OpenAI" in inf_src)
check("inference.py: reads API_BASE_URL", "API_BASE_URL" in inf_src)
check("inference.py: reads MODEL_NAME", "MODEL_NAME" in inf_src)
check("inference.py: reads HF_TOKEN", "HF_TOKEN" in inf_src)
check("inference.py: reads ENV_BASE_URL", "ENV_BASE_URL" in inf_src)


# Check README frontmatter
with open(os.path.join(ROOT, "README.md")) as f:
   readme = f.read()
check("README: has sdk: docker", "sdk: docker" in readme)
check("README: has tags openenv", "openenv" in readme)
check("README: has action space section", "Action Space" in readme or "action_type" in readme)
check("README: has baseline scores", "Baseline" in readme)


# Check Dockerfile correct port
with open(os.path.join(ROOT, "Dockerfile")) as f:
   dockerfile = f.read()
check("Dockerfile: port 7860", "7860" in dockerfile)
check("Dockerfile: python:3.11-slim", "python:3.11-slim" in dockerfile)


# Check openenv.yaml
import yaml
with open(os.path.join(ROOT, "openenv.yaml")) as f:
   oe = yaml.safe_load(f)
check("openenv.yaml: has name", "name" in oe)
check("openenv.yaml: has version", "version" in oe)
check("openenv.yaml: has description", "description" in oe)
check("openenv.yaml: has tags with openenv", "openenv" in oe.get("tags", []))
check("openenv.yaml: has tasks (>=3)", len(oe.get("tasks", [])) >= 3)
check("openenv.yaml: has action_space", "action_space" in oe)
check("openenv.yaml: has observation_space", "observation_space" in oe)
check("openenv.yaml: has reward", "reward" in oe)


# Check each task has required fields
for t in oe.get("tasks", []):
   tid = t.get("id", "?")
   check(f"openenv.yaml: task '{tid}' has id", "id" in t)
   check(f"openenv.yaml: task '{tid}' has description", "description" in t)
   check(f"openenv.yaml: task '{tid}' has difficulty", "difficulty" in t)
   check(f"openenv.yaml: task '{tid}' has max_score", "max_score" in t)




# ===================================================================
# SECTION 4: WebSocket test
# ===================================================================
print("\n" + "=" * 65)
print("  SECTION 4: WebSocket test")
print("=" * 65)


try:
   import websockets
   import asyncio


   async def ws_test():
       async with websockets.connect("ws://localhost:7860/ws") as ws:
           # Reset
           await ws.send(json.dumps({"type": "reset", "task_id": "easy"}))
           resp = json.loads(await ws.recv())
           check("ws: reset response type=step_result", resp.get("type") == "step_result")
           check("ws: reset done=False", resp.get("done") == False)


           # Step
           await ws.send(json.dumps({
               "type": "step",
               "action": {"action_type": "inspect_function", "function_name": "calculate_average"}
           }))
           resp = json.loads(await ws.recv())
           check("ws: step response type=step_result", resp.get("type") == "step_result")
           check("ws: step reward=0.0", resp.get("reward") == 0.0)


           # State
           await ws.send(json.dumps({"type": "state"}))
           resp = json.loads(await ws.recv())
           check("ws: state response type=state", resp.get("type") == "state")
           check("ws: state has episode_id", "episode_id" in resp)


           # Unknown type
           await ws.send(json.dumps({"type": "foobar"}))
           resp = json.loads(await ws.recv())
           check("ws: unknown type returns error", resp.get("type") == "error")


   asyncio.run(ws_test())
except Exception as e:
   print(f"  ⚠️ WebSocket test error: {e}")
   traceback.print_exc()




# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 65)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
   print("  🎉 ALL CHECKS PASSED!")
else:
   print(f"  ⚠️  {FAIL} checks FAILED — review above")
print("=" * 65)


sys.exit(0 if FAIL == 0 else 1)