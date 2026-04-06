"""
FastAPI server for BugHunt.


Endpoints (OpenEnv spec):
   GET  /health
   POST /reset?task_id=easy
   POST /step
   GET  /state
   WS   /ws      ← persistent session (used by OpenEnv clients)
   GET  /docs
   GET  /web
"""
from __future__ import annotations
import json


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, RedirectResponse


from models import BugHuntAction
from environment import BugHuntEnvironment


app = FastAPI(
   title="BugHunt",
   description=(
       "An OpenEnv RL environment where agents debug Python code by inspecting "
       "functions, running tests, and proposing fixes. Three tasks: easy → medium → hard."
   ),
   version="1.0.0",
)


# Shared stateless HTTP environment
_http_env = BugHuntEnvironment()




@app.get("/")
def root():
   return RedirectResponse(url="/web")




@app.get("/health")
def health():
   return {"status": "healthy", "environment": "bughunt", "version": "1.0.0"}




@app.post("/reset")
def reset(task_id: str = Query(default="easy", description="easy | medium | hard")):
   obs = _http_env.reset(task_id=task_id)
   return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}




@app.post("/step")
def step(action: BugHuntAction):
   obs = _http_env.step(action)
   return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}




@app.get("/state")
def state():
   return _http_env.state.model_dump()




# ------------------------------------------------------------------
# WebSocket — one isolated environment per connection
# ------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
   env = BugHuntEnvironment()
   await websocket.accept()
   try:
       while True:
           raw = await websocket.receive_text()
           data = json.loads(raw)
           msg_type = data.get("type", "")


           if msg_type == "reset":
               obs = env.reset(
                   task_id=data.get("task_id", "easy"),
                   episode_id=data.get("episode_id"),
               )
               await websocket.send_text(json.dumps({
                   "type": "step_result",
                   "observation": obs.model_dump(),
                   "reward": obs.reward,
                   "done": obs.done,
               }))


           elif msg_type == "step":
               action = BugHuntAction(**data.get("action", {}))
               obs = env.step(action)
               await websocket.send_text(json.dumps({
                   "type": "step_result",
                   "observation": obs.model_dump(),
                   "reward": obs.reward,
                   "done": obs.done,
               }))


           elif msg_type == "state":
               await websocket.send_text(json.dumps({
                   "type": "state",
                   **env.state.model_dump(),
               }))


           else:
               await websocket.send_text(json.dumps({
                   "type": "error",
                   "message": f"Unknown type '{msg_type}'. Use reset | step | state.",
               }))


   except WebSocketDisconnect:
       pass
   except Exception as exc:
       try:
           await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
       except Exception:
           pass




# ------------------------------------------------------------------
# Web UI
# ------------------------------------------------------------------


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BugHunt — OpenEnv Debugging Environment</title>
<style>
 body{font-family:monospace;max-width:960px;margin:40px auto;padding:20px;background:#1e1e1e;color:#d4d4d4}
 h1{color:#4ec9b0}h2{color:#9cdcfe}
 code{background:#2d2d2d;padding:2px 6px;border-radius:3px}
 pre{background:#2d2d2d;padding:16px;border-radius:6px;overflow-x:auto}
 a{color:#4fc1ff}
 .badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:12px;margin:2px}
 .easy{background:#1a4a1a;color:#4ec9b0}
 .medium{background:#4a3a1a;color:#ce9178}
 .hard{background:#4a1a1a;color:#f48771}
</style>
</head>
<body>
<h1>🔍 BugHunt</h1>
<p>An OpenEnv RL environment where agents find and fix bugs in Python code.</p>


<h2>Tasks</h2>
<p>
 <span class="badge easy">EASY</span> 1 bug · 5 tests · stats module<br>
 <span class="badge medium">MEDIUM</span> 2 bugs · 7 tests · text processor<br>
 <span class="badge hard">HARD</span> 3 bugs (2 interdependent) · 9 tests · grade calculator
</p>


<h2>Actions</h2>
<pre>
inspect_function  {"action_type":"inspect_function","function_name":"calculate_average"}
run_test          {"action_type":"run_test","test_id":"E1"}
propose_fix       {"action_type":"propose_fix","function_name":"calculate_average",
                  "new_code":"def calculate_average(numbers):\\n    if not numbers:\\n        return 0\\n    return sum(numbers) / len(numbers)"}
submit            {"action_type":"submit"}
</pre>


<h2>Quick Start</h2>
<pre>
# 1. Reset
curl -X POST "/reset?task_id=easy"


# 2. Inspect a function
curl -X POST "/step" -H "Content-Type: application/json" \
 -d '{"action_type":"inspect_function","function_name":"calculate_average"}'


# 3. Run a test
curl -X POST "/step" -d '{"action_type":"run_test","test_id":"E1"}'


# 4. Fix it
curl -X POST "/step" -H "Content-Type: application/json" \
 -d '{"action_type":"propose_fix","function_name":"calculate_average",
      "new_code":"def calculate_average(numbers):\\n    if not numbers:\\n        return 0\\n    return sum(numbers) / len(numbers)"}'


# 5. Submit
curl -X POST "/step" -d '{"action_type":"submit"}'
</pre>


<p>Full API: <a href="/docs">/docs</a></p>
</body>
</html>
"""




@app.get("/web", response_class=HTMLResponse)
def web_ui():
   return _HTML