#!/usr/bin/env python3
"""
BugHunt — Live HTTP Integration Tests

Tests the DEPLOYED HF Space via HTTP, exactly as judges will.
"""
import requests
import json
import sys
import time

BASE = "https://ayushxx9-bughunt-env.hf.space"
SESSION = "judge_test_session"
HEADERS = {"Content-Type": "application/json", "x-session-id": SESSION}
PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


def reset(task_id="easy"):
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def step(action):
    r = requests.post(f"{BASE}/step", json={"action": action}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get(endpoint):
    r = requests.get(f"{BASE}{endpoint}", timeout=15)
    r.raise_for_status()
    return r.json()


print("=" * 60)
print("🔍 BugHunt — Live HTTP Judge Test Suite")
print(f"   Target: {BASE}")
print("=" * 60)

# ── 1. Health & Metadata ──────────────────────────────────────────
print("\n📋 1. Health & Metadata")
health = get("/health")
check("Health endpoint", health.get("status") == "healthy", str(health))

meta = get("/metadata")
check("Metadata has name", "name" in meta, str(meta))

schema = get("/schema")
check("Schema has action", "action" in schema)
check("Schema has observation", "observation" in schema)
check("Schema has state", "state" in schema)

caps = get("/env/capabilities")
check("Capabilities version", caps.get("version") == "2.0.0")
check("Has interdependent_bugs feature", caps.get("features", {}).get("interdependent_bugs") is True)

# ── 2. Full RL Loop: Easy ─────────────────────────────────────────
print("\n🎮 2. Full RL Loop (Easy)")
result = reset("easy")
obs = result["observation"]
check("Reset returns observation", "task_id" in obs)
check("Initial score is 0.4", abs(obs["current_score"] - 0.4) < 0.01, f"got {obs['current_score']}")
check("Done is False", result["done"] is False)
check("Has 5 tests", obs["tests_total"] == 5, f"got {obs['tests_total']}")
check("Has 2 functions", len(obs["available_functions"]) == 2)

# Inspect
result = step({"action_type": "inspect_function", "function_name": "calculate_average"})
obs = result["observation"]
check("Inspect is free (reward=0)", result["reward"] == 0.0, f"reward={result['reward']}")
check("Source code returned", "calculate_average" in obs.get("inspected_functions", {}))
src = obs["inspected_functions"].get("calculate_average", "")
check("Source shows bug", "len(numbers) - 1" in src, f"code={src[:80]}")

# Run test
result = step({"action_type": "run_test", "test_id": "E1"})
check("Run test is free", result["reward"] == 0.0)
e1 = next(r for r in result["observation"]["test_results"] if r["test_id"] == "E1")
check("E1 fails (has bug)", e1["status"] == "fail", f"status={e1['status']}")

result = step({"action_type": "run_test", "test_id": "E5"})
e5 = next(r for r in result["observation"]["test_results"] if r["test_id"] == "E5")
check("E5 passes (no bug)", e5["status"] == "pass", f"status={e5['status']}")

# Fix
fix = "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n"
result = step({"action_type": "propose_fix", "function_name": "calculate_average", "new_code": fix})
check("Good fix gives positive reward", result["reward"] > 0, f"reward={result['reward']}")
check("Score now 1.0", abs(result["observation"]["current_score"] - 1.0) < 0.01,
      f"score={result['observation']['current_score']}")

# Submit
result = step({"action_type": "submit"})
check("Submit returns done=True", result["done"] is True)
check("Submit reward = final score", result["reward"] is not None and result["reward"] >= 0)

# ── 3. Full RL Loop: Hard (Interdependency Proof) ────────────────
print("\n🔗 3. Hard Mode Interdependency Test")
result = reset("hard")
obs = result["observation"]
check("Hard has 9 tests", obs["tests_total"] == 9)
check("Hard has 4 functions", len(obs["available_functions"]) == 4)

# Fix ONLY Bug 2 (swapped args) — should NOT improve
fix_bug2_only = ("def calculate_final_grade(assignments, midterm, final_exam, weights=(0.3,0.3,0.4)):\n"
                 "    if not assignments:\n"
                 "        return 'F'\n"
                 "    avg = sum(assignments)/len(assignments)\n"
                 "    composite = weighted_average([avg, midterm, final_exam], list(weights))\n"
                 "    return letter_grade(composite)\n")
score_before = obs["current_score"]
result = step({"action_type": "propose_fix", "function_name": "calculate_final_grade", "new_code": fix_bug2_only})
score_after = result["observation"]["current_score"]
check("🔑 Bug2 alone gives NO improvement", score_after <= score_before + 0.001,
      f"BROKEN! Score went {score_before}→{score_after}")
check("🔑 Bug2 alone gives penalty", result["reward"] <= 0, f"reward={result['reward']}")

# Now fix Bug 1 (arithmetic) — now BOTH should work
fix_bug1 = ("def weighted_average(scores, weights):\n"
            "    if len(scores)!=len(weights):\n"
            "        return 0\n"
            "    total=0\n"
            "    for score,weight in zip(scores,weights):\n"
            "        total+=score*weight\n"
            "    return total\n")
result = step({"action_type": "propose_fix", "function_name": "weighted_average", "new_code": fix_bug1})
check("🔑 Bug1+Bug2 together gives positive reward", result["reward"] > 0, f"reward={result['reward']}")
check("🔑 Score improved significantly", result["observation"]["current_score"] > score_before + 0.2,
      f"score={result['observation']['current_score']}")

# Fix Bug 3 (independent)
fix_bug3 = ("def class_statistics(student_scores):\n"
            "    if not student_scores:\n"
            "        return {'mean':0,'highest':0,'lowest':0,'passing':0}\n"
            "    scores = list(student_scores.values())\n"
            "    passing = sum(1 for s in scores if s >= 60)\n"
            "    return {'mean':round(sum(scores)/len(scores),2),'highest':max(scores),'lowest':min(scores),'passing':passing}\n")
result = step({"action_type": "propose_fix", "function_name": "class_statistics", "new_code": fix_bug3})
check("Bug3 independent fix works", result["reward"] > 0, f"reward={result['reward']}")

result = step({"action_type": "submit"})
check("Hard full solve = 1.0", abs(result["reward"] - 1.0) < 0.01, f"reward={result['reward']}")

# ── 4. Edge Cases Over HTTP ──────────────────────────────────────
print("\n⚠️  4. Edge Cases (HTTP)")

# Step after done
result = step({"action_type": "inspect_function", "function_name": "weighted_average"})
check("Step after done returns done=True", result["done"] is True)

# Reset clears state
result = reset("easy")
check("Reset clears previous episode", result["observation"]["current_score"] < 0.5)
check("Reset clears ops log", len(result["observation"]["operations_log"]) == 0)

# Invalid action
result = step({"action_type": "garbage_action"})
check("Invalid action gives penalty", result["reward"] == -0.05, f"reward={result['reward']}")

# Nonexistent function
result = step({"action_type": "inspect_function", "function_name": "nonexistent"})
check("Bad function name gives penalty", result["reward"] == -0.05)

# Nonexistent test
result = step({"action_type": "run_test", "test_id": "Z99"})
check("Bad test id gives penalty", result["reward"] == -0.05)

# Sandbox: import blocked
result = step({"action_type": "propose_fix", "function_name": "calculate_average",
               "new_code": "def calculate_average(n):\n    import os\n    return 0\n"})
check("Import blocked", result["reward"] == -0.05)

# Sandbox: eval blocked
result = step({"action_type": "propose_fix", "function_name": "calculate_average",
               "new_code": "def calculate_average(n):\n    return eval('1')\n"})
check("Eval blocked", result["reward"] == -0.05)

# ── 5. Custom Endpoints ──────────────────────────────────────────
print("\n📊 5. Custom Endpoints")
analytics = get("/analytics")
check("Analytics returns total_episodes", "total_episodes" in analytics)
check("Analytics has uptime", "uptime_s" in analytics)

curriculum = get("/curriculum")
check("Curriculum has current_difficulty", "current_difficulty" in curriculum)
check("Curriculum starts at easy", curriculum["current_difficulty"] == "easy")

tasks = get("/tasks/info")
check("Tasks info has 3 tasks", len(tasks.get("tasks", {})) == 3)

dep = get("/tasks/dependency_graph/hard")
check("Dep graph has 3 bugs", len(dep.get("bugs", [])) == 3)
check("Dep graph has dependencies", len(dep.get("dependencies", [])) > 0)
check("Has optimal strategy", "optimal_strategy" in dep)

leaderboard = get("/leaderboard")
check("Leaderboard is dict", isinstance(leaderboard, dict))

# ── 6. Determinism over HTTP ─────────────────────────────────────
print("\n🔁 6. Determinism (HTTP)")
scores_a = []
scores_b = []
for _ in range(3):
    r = reset("easy")
    scores_a.append(r["observation"]["current_score"])
for _ in range(3):
    r = reset("hard")
    scores_b.append(r["observation"]["current_score"])
check("Easy initial score deterministic", len(set(f"{s:.4f}" for s in scores_a)) == 1,
      f"scores={scores_a}")
check("Hard initial score deterministic", len(set(f"{s:.4f}" for s in scores_b)) == 1,
      f"scores={scores_b}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed")
if FAIL:
    print(f"\n  ❌ FAILURES ({FAIL}):")
    for e in ERRORS:
        print(f"     • {e}")
else:
    print("  🏆 ALL TESTS PASSED — Ready for judging!")
print("=" * 60)

sys.exit(1 if FAIL else 0)
