# server/gradio_ui.py
"""
Custom Gradio UI for BugHunt — Interactive debugging playground.

This provides an interactive web interface at /web where humans
can manually play the BugHunt environment, inspect code, run tests,
propose fixes, and see real-time scoring visualizations.

Used as the `gradio_builder` argument in create_app().
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import gradio as gr


# ── CSS for premium dark UI ──────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bh-bg: #0a0a0f;
    --bh-surface: #12121a;
    --bh-surface2: #1a1a28;
    --bh-border: #2a2a3d;
    --bh-accent: #7c3aed;
    --bh-accent2: #a855f7;
    --bh-green: #10b981;
    --bh-red: #ef4444;
    --bh-yellow: #f59e0b;
    --bh-blue: #3b82f6;
    --bh-text: #e2e8f0;
    --bh-text-dim: #94a3b8;
    --bh-code-bg: #0d1117;
}

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bh-bg) !important;
    max-width: 1400px !important;
}

#bughunt-header {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    padding: 1rem 0;
    font-family: 'Inter', sans-serif;
}

.bh-card {
    background: var(--bh-surface) !important;
    border: 1px solid var(--bh-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.score-display {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}

.test-pass { color: var(--bh-green) !important; }
.test-fail { color: var(--bh-red) !important; }
.test-notrun { color: var(--bh-text-dim) !important; }

.action-btn {
    background: linear-gradient(135deg, var(--bh-accent), var(--bh-accent2)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
}

.submit-btn {
    background: linear-gradient(135deg, var(--bh-green), #059669) !important;
}

.code-block {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bh-code-bg) !important;
    border: 1px solid var(--bh-border) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

.metric-card {
    background: var(--bh-surface2) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    text-align: center !important;
}

.difficulty-easy { border-left: 3px solid var(--bh-green) !important; }
.difficulty-medium { border-left: 3px solid var(--bh-yellow) !important; }
.difficulty-hard { border-left: 3px solid var(--bh-red) !important; }
"""

# ── State management ─────────────────────────────────────────────────────────

_session_state: Dict[str, Any] = {
    "env": None,
    "obs": None,
    "history": [],
    "start_time": None,
    "reward_trace": [],
}


def _format_score_html(score: float, tests_passed: int, tests_total: int) -> str:
    """Generate beautiful score display HTML."""
    pct = int(score * 100)
    color = "#10b981" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
    return f"""
    <div style="text-align:center; padding:1rem;">
        <div style="font-size:3.5rem; font-weight:800; color:{color};
                    font-family:'JetBrains Mono',monospace;
                    text-shadow: 0 0 20px {color}40;">
            {pct}%
        </div>
        <div style="font-size:1rem; color:#94a3b8; margin-top:0.25rem;">
            {tests_passed}/{tests_total} tests passing
        </div>
        <div style="width:100%; height:6px; background:#1e1e2e;
                    border-radius:3px; margin-top:0.75rem; overflow:hidden;">
            <div style="width:{pct}%; height:100%;
                        background:linear-gradient(90deg, {color}, {color}80);
                        border-radius:3px; transition:width 0.5s ease;"></div>
        </div>
    </div>
    """


def _format_test_results_html(test_results: list) -> str:
    """Generate HTML for test results with status icons."""
    rows = []
    for tr in test_results:
        status = tr.get("status", "not_run")
        icon_map = {
            "pass": ("✅", "#10b981", "PASS"),
            "fail": ("❌", "#ef4444", "FAIL"),
            "error": ("💥", "#f59e0b", "ERROR"),
            "not_run": ("⬜", "#64748b", "PENDING"),
        }
        icon, color, label = icon_map.get(status, ("⬜", "#64748b", "?"))
        hint = f'<span style="color:#94a3b8;font-size:0.8rem"> — {tr["output"]}</span>' if tr.get("output") else ""
        rows.append(f"""
        <div style="display:flex; align-items:center; gap:0.5rem; padding:0.5rem 0.75rem;
                    background:#12121a; border-radius:6px; margin-bottom:0.25rem;
                    border-left:3px solid {color};">
            <span style="font-size:1.1rem;">{icon}</span>
            <span style="color:{color}; font-weight:600; font-size:0.75rem;
                         font-family:'JetBrains Mono',monospace; min-width:45px;">{label}</span>
            <span style="color:#e2e8f0; font-size:0.85rem;">
                <b>[{tr['test_id']}]</b> {tr['description'][:60]}
            </span>
            {hint}
        </div>
        """)
    return '<div style="display:flex; flex-direction:column; gap:2px;">' + "".join(rows) + "</div>"


def _format_ops_log_html(log: list) -> str:
    """Format operations log."""
    if not log:
        return '<div style="color:#64748b;text-align:center;padding:1rem;">No operations yet</div>'
    entries = []
    for i, op in enumerate(log[-10:]):
        entries.append(f"""
        <div style="display:flex; gap:0.5rem; padding:0.35rem 0.5rem;
                    font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                    color:#94a3b8; border-left:2px solid #7c3aed40;">
            <span style="color:#7c3aed; min-width:20px;">#{i+1}</span>
            <span style="color:#e2e8f0;">{op}</span>
        </div>
        """)
    return "".join(entries)


def _format_reward_chart(reward_trace: list) -> str:
    """Generate a text-based reward chart."""
    if not reward_trace:
        return '<div style="color:#64748b;text-align:center;">No rewards yet</div>'
    bars = []
    for i, (action, reward) in enumerate(reward_trace):
        if reward is None:
            continue
        width = max(5, abs(reward) * 200)
        color = "#10b981" if reward > 0 else "#ef4444" if reward < 0 else "#3b82f6"
        bars.append(f"""
        <div style="display:flex; align-items:center; gap:0.5rem; margin:2px 0;">
            <span style="color:#94a3b8; font-size:0.7rem; min-width:20px; text-align:right;">
                {i+1}
            </span>
            <div style="width:{width}px; height:12px; background:{color};
                        border-radius:2px; opacity:0.8;"></div>
            <span style="color:{color}; font-family:'JetBrains Mono',monospace;
                         font-size:0.75rem;">{reward:+.3f}</span>
            <span style="color:#64748b; font-size:0.7rem;">{action}</span>
        </div>
        """)
    return "".join(bars)


def build_bughunt_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """
    Custom Gradio UI builder for BugHunt.

    This replaces the default OpenEnv Gradio interface with a beautiful,
    purpose-built debugging playground.
    """
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="🔍 BugHunt — AI Debugging Arena",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    ) as demo:
        # Header
        gr.HTML("""
        <div style="text-align:center; padding:1.5rem 0 0.5rem;">
            <div style="font-size:2.8rem; font-weight:800;
                        background:linear-gradient(135deg, #7c3aed, #3b82f6, #10b981);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🔍 BugHunt
            </div>
            <div style="color:#94a3b8; font-size:1rem; margin-top:0.25rem;">
                AI Debugging Arena — Find & Fix Python Bugs Through RL
            </div>
            <div style="display:flex; justify-content:center; gap:1rem; margin-top:0.75rem;">
                <span style="background:#7c3aed20; color:#a855f7; padding:0.25rem 0.75rem;
                             border-radius:20px; font-size:0.75rem; font-weight:600;">
                    OpenEnv SDK v2
                </span>
                <span style="background:#3b82f620; color:#60a5fa; padding:0.25rem 0.75rem;
                             border-radius:20px; font-size:0.75rem; font-weight:600;">
                    3 Difficulty Levels
                </span>
                <span style="background:#10b98120; color:#34d399; padding:0.25rem 0.75rem;
                             border-radius:20px; font-size:0.75rem; font-weight:600;">
                    Interdependent Bugs
                </span>
            </div>
        </div>
        """)

        # ── Control bar ──
        with gr.Row():
            task_select = gr.Radio(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="🎯 Difficulty",
                info="Easy=1 bug, Medium=2 bugs, Hard=3 bugs (2 interdependent)",
            )
            reset_btn = gr.Button("🔄 New Episode", variant="primary", scale=0)

        # ── State displays ──
        with gr.Row():
            score_html = gr.HTML(
                _format_score_html(0, 0, 0),
                label="Score",
            )
            message_box = gr.Textbox(
                label="📋 Environment Message",
                interactive=False,
                lines=2,
            )

        with gr.Row():
            ops_remaining = gr.Number(label="⚡ Ops Remaining", interactive=False, value=0)
            step_count = gr.Number(label="👣 Steps Taken", interactive=False, value=0)

        # ── Main panels ──
        with gr.Tabs():
            # ── TAB: Inspect & Run ──
            with gr.Tab("🔎 Inspect & Test"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 Inspect Functions")
                        fn_name = gr.Dropdown(
                            label="Function", choices=[], interactive=True
                        )
                        inspect_btn = gr.Button("🔍 Inspect", elem_classes=["action-btn"])
                        source_code = gr.Code(
                            label="Source Code",
                            language="python",
                            interactive=False,
                            lines=12,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🧪 Run Tests")
                        test_select = gr.Dropdown(
                            label="Test Case", choices=[], interactive=True
                        )
                        run_test_btn = gr.Button("▶️ Run Test", elem_classes=["action-btn"])
                        test_results_html = gr.HTML(label="Test Results")

            # ── TAB: Propose Fix ──
            with gr.Tab("🛠️ Fix Code"):
                gr.Markdown("### Propose a Fix")
                gr.Markdown(
                    "*Paste your corrected function below. "
                    "Must start with `def `. No imports allowed.*"
                )
                fix_fn_name = gr.Dropdown(
                    label="Function to Fix", choices=[], interactive=True
                )
                fix_code = gr.Code(
                    label="New Code",
                    language="python",
                    interactive=True,
                    lines=12,
                    value="def calculate_average(numbers):\n    pass\n",
                )
                propose_btn = gr.Button(
                    "🚀 Propose Fix", variant="primary", elem_classes=["action-btn"]
                )

            # ── TAB: Submit & Analytics ──
            with gr.Tab("📊 Submit & Analytics"):
                submit_btn = gr.Button(
                    "📤 Submit Solution",
                    variant="primary",
                    elem_classes=["submit-btn"],
                )
                with gr.Row():
                    reward_chart_html = gr.HTML(label="Reward Trace")
                    ops_log_html = gr.HTML(label="Operations Log")

            # ── TAB: Info ──
            with gr.Tab("ℹ️ How to Play"):
                gr.Markdown("""
## 🎮 How to Play BugHunt

**Goal:** Find and fix bugs in Python functions to make all tests pass.

### Actions (in order):
1. **🔍 Inspect** — Read the source code of each function (free, no penalty)
2. **🧪 Run Tests** — See which tests fail and read the hints (free)
3. **🛠️ Propose Fix** — Submit corrected code (rewarded if score improves!)
4. **📤 Submit** — Finalize your solution

### Reward System:
| Action | Reward |
|--------|--------|
| Inspect / Run Test | 0.0 (free) |
| Fix that improves score | +Δscore |
| Fix with no improvement | -0.05 |
| Submit | final score [0-1] |

### The Hard Mode Challenge 🔗
Two of three bugs are **interdependent** — fixing one without the other
yields ZERO improvement. You must reason about the coupling!

### Tips:
- Always inspect ALL functions first
- Run ALL tests to see the failure patterns
- Read the hints carefully — they tell you exactly what's wrong
- On hard mode, fix Bug 1 (arithmetic) BEFORE or WITH Bug 2 (order)
                """)

        # ── Action handlers ──

        def do_reset(task_id):
            """Reset environment with selected task."""
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                web_manager.reset_environment({"task_id": task_id})
            ) if hasattr(web_manager, 'reset_environment') else _sync_reset(web_manager, task_id)

            obs = result.get("observation", result)
            _session_state["start_time"] = time.time()
            _session_state["reward_trace"] = []
            _session_state["obs"] = obs

            return _update_ui(obs, result.get("reward"), result.get("done", False))

        def _sync_reset(wm, task_id):
            """Synchronous reset fallback."""
            env = wm.env if hasattr(wm, 'env') else wm
            obs = env.reset(task_id=task_id)
            return {
                "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
                "reward": obs.reward if hasattr(obs, 'reward') else None,
                "done": obs.done if hasattr(obs, 'done') else False,
            }

        def do_inspect(fn):
            """Inspect a function."""
            if not fn:
                return [gr.update()] * 8
            result = _sync_step(web_manager, {
                "action_type": "inspect_function",
                "function_name": fn,
            })
            obs = result.get("observation", {})
            _session_state["reward_trace"].append(("inspect", result.get("reward", 0)))
            code = obs.get("inspected_functions", {}).get(fn, "")
            updates = _update_ui(obs, result.get("reward"), result.get("done", False))
            return [code] + list(updates)

        def do_run_test(test_id):
            """Run a test case."""
            if not test_id:
                return [gr.update()] * 7
            result = _sync_step(web_manager, {
                "action_type": "run_test",
                "test_id": test_id,
            })
            obs = result.get("observation", {})
            _session_state["reward_trace"].append(("test", result.get("reward", 0)))
            return _update_ui(obs, result.get("reward"), result.get("done", False))

        def do_propose(fn, code):
            """Propose a fix."""
            if not fn or not code:
                return [gr.update()] * 7
            result = _sync_step(web_manager, {
                "action_type": "propose_fix",
                "function_name": fn,
                "new_code": code,
            })
            obs = result.get("observation", {})
            _session_state["reward_trace"].append(("fix", result.get("reward", 0)))
            return _update_ui(obs, result.get("reward"), result.get("done", False))

        def do_submit():
            """Submit solution."""
            result = _sync_step(web_manager, {"action_type": "submit"})
            obs = result.get("observation", {})
            _session_state["reward_trace"].append(("submit", result.get("reward", 0)))
            return _update_ui(obs, result.get("reward"), result.get("done", False))

        def _sync_step(wm, action_data):
            """Synchronous step."""
            env = wm.env if hasattr(wm, 'env') else wm
            from models import BugHuntAction
            action = BugHuntAction(**action_data)
            obs = env.step(action)
            return {
                "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
                "reward": obs.reward if hasattr(obs, 'reward') else None,
                "done": obs.done if hasattr(obs, 'done') else False,
            }

        def _update_ui(obs, reward, done):
            """Update all UI components from observation."""
            score = obs.get("current_score", 0)
            tp = obs.get("tests_passed", 0)
            tt = obs.get("tests_total", 0)
            msg = obs.get("message", "")
            if done:
                elapsed = time.time() - (_session_state.get("start_time") or time.time())
                msg = f"🏁 {msg} (took {elapsed:.1f}s)"

            fns = obs.get("available_functions", [])
            tests = obs.get("available_tests", [])

            return (
                _format_score_html(score, tp, tt),
                msg,
                obs.get("operations_remaining", 0),
                obs.get("step_count", len(obs.get("operations_log", []))),
                gr.update(choices=fns, value=fns[0] if fns else None),
                gr.update(choices=tests, value=tests[0] if tests else None),
                _format_test_results_html(obs.get("test_results", [])),
            )

        # Wire events — reset returns 7 UI outputs + dropdowns for fix tab
        main_outputs = [
            score_html, message_box, ops_remaining, step_count,
            fn_name, test_select, test_results_html,
        ]

        reset_btn.click(
            do_reset,
            inputs=[task_select],
            outputs=main_outputs,
        ).then(
            lambda obs: gr.update(choices=obs if isinstance(obs, list) else []),
            inputs=[fn_name],
            outputs=[fix_fn_name],
        )

        inspect_btn.click(
            do_inspect,
            inputs=[fn_name],
            outputs=[source_code] + main_outputs,
        )

        run_test_btn.click(
            do_run_test,
            inputs=[test_select],
            outputs=main_outputs,
        )

        propose_btn.click(
            do_propose,
            inputs=[fix_fn_name, fix_code],
            outputs=main_outputs,
        )

        submit_btn.click(
            do_submit,
            outputs=main_outputs,
        ).then(
            lambda: (
                _format_reward_chart(_session_state.get("reward_trace", [])),
                _format_ops_log_html(
                    _session_state.get("obs", {}).get("operations_log", [])
                    if isinstance(_session_state.get("obs"), dict)
                    else []
                ),
            ),
            outputs=[reward_chart_html, ops_log_html],
        )

    return demo
