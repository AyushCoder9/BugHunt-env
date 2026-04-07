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


import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
  :root { --bg: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --accent-glow: rgba(34, 211, 238, 0.5); --text-main: #f8fafc; --text-muted: #94a3b8; }
  body { font-family: 'Inter', sans-serif; background: #0f172a; color: var(--text-main); margin: 0; padding: 40px 20px; line-height: 1.6; }
  .container { max-width: 1000px; margin: 0 auto; position: relative; }
  
  /* Glow effects */
  .glow-circle { position: absolute; width: 400px; height: 400px; background: radial-gradient(circle, rgba(168,85,247,0.15) 0%, rgba(15,23,42,0) 70%); top: -100px; left: -100px; z-index: -1; pointer-events: none; }
  .glow-circle-2 { position: absolute; width: 500px; height: 500px; background: radial-gradient(circle, rgba(34,211,238,0.1) 0%, rgba(15,23,42,0) 70%); bottom: 0; right: -200px; z-index: -1; pointer-events: none; }
  
  header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; margin-bottom: 40px; }
  h1 { font-weight: 800; font-size: 2.5rem; margin: 0; background: linear-gradient(to right, #22d3ee, #a855f7); -webkit-background-clip: text; color: transparent; letter-spacing: -1px; }
  .subtitle { color: var(--text-muted); font-size: 1.1rem; margin-top: 5px; }
  
  .status-badge { display: flex; align-items: center; gap: 8px; background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.2); padding: 8px 16px; border-radius: 999px; color: #4ade80; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; }
  .status-badge:hover { background: rgba(74, 222, 128, 0.2); box-shadow: 0 0 15px rgba(74,222,128,0.3); }
  .pulse { width: 8px; height: 8px; background-color: #4ade80; border-radius: 50%; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); animation: pulse 2s infinite; }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); } 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); } }
  
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-bottom: 40px; }
  .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; transition: transform 0.2s, border-color 0.2s; }
  .card:hover { transform: translateY(-3px); border-color: rgba(255,255,255,0.15); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); }
  
  .tag { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px; }
  .tag.easy { background: rgba(74,222,128,0.15); color: #4ade80; }
  .tag.medium { background: rgba(250,204,21,0.15); color: #facc15; }
  .tag.hard { background: rgba(248,113,113,0.15); color: #f87171; }
  
  .card h3 { margin: 0 0 10px 0; font-size: 1.25rem; }
  .card p { margin: 0; color: var(--text-muted); font-size: 0.9rem; }
  .stats { margin-top: 15px; display: flex; gap: 15px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b; }
  
  .terminal { background: #020617; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38bdf8; position: relative; }
  .terminal-header { display: flex; gap: 8px; margin-bottom: 15px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot.r { background: #ef4444; } .dot.y { background: #eab308; } .dot.g { background: #22c55e; }
  
  .docs-link { display: inline-block; margin-top: 30px; color: #a855f7; text-decoration: none; font-weight: 600; border-bottom: 1px dashed #a855f7; padding-bottom: 2px; transition: color 0.2s; }
  .docs-link:hover { color: #d8b4fe; border-bottom-style: solid; }
  
  #api-response { white-space: pre-wrap; margin: 0; color: #a3e635; }
</style>
</head>
<body>
<div class="glow-circle"></div>
<div class="glow-circle-2"></div>
<div class="container">
  <header>
    <div>
      <h1>BugHunt</h1>
      <div class="subtitle">OpenEnv Reinforcement Learning Sandbox</div>
    </div>
    <div class="status-badge" onclick="pingHealth()" title="Click to ping API health">
      <div class="pulse"></div>
      Environment Live
    </div>
  </header>
  
  <div class="grid">
    <div class="card">
      <div class="tag easy">Easy</div>
      <h3>stats_module</h3>
      <p>Fix an off-by-one error causing a ZeroDivisionError in average calculation.</p>
      <div class="stats"><span>1 Bug</span><span>5 Tests</span><span>10 Ops</span></div>
    </div>
    <div class="card">
      <div class="tag medium">Medium</div>
      <h3>text_processor</h3>
      <p>Fix two independent logic errors in string manipulation and text slicing.</p>
      <div class="stats"><span>2 Bugs</span><span>7 Tests</span><span>15 Ops</span></div>
    </div>
    <div class="card" style="border-color: rgba(248,113,113,0.3);">
      <div class="tag hard">Hard</div>
      <h3>grade_calculator</h3>
      <p>Identify and patch three bugs, including a complex procedurally interdependent arithmetic error.</p>
      <div class="stats"><span>3 Bugs</span><span>9 Tests</span><span>20 Ops</span></div>
    </div>
  </div>

  <div class="terminal">
    <div class="terminal-header">
      <div class="dot r"></div><div class="dot y"></div><div class="dot g"></div>
    </div>
    <pre id="api-response">> Awaiting API interaction...
> Try clicking the "Environment Live" button above to ping /health</pre>
  </div>
  
  <a href="/docs" class="docs-link">View Full Swagger API Documentation &rarr;</a>
</div>

<script>
  async function pingHealth() {
    const resBox = document.getElementById('api-response');
    resBox.innerText = '> Fetching /health...\n';
    try {
      const response = await fetch('/health');
      const data = await response.json();
      resBox.innerText += '> Response: 200 OK\n';
      resBox.innerText += JSON.stringify(data, null, 2);
    } catch (e) {
      resBox.innerText += '> Error reaching host: ' + e.message;
      resBox.style.color = '#ef4444';
    }
  }
</script>
</body>
</html>"""




import uvicorn

@app.get("/web", response_class=HTMLResponse)
def web_ui():
   return _HTML

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()