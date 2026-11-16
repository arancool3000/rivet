#!/usr/bin/env python3
"""Scheduler tools (MCP over stdio): the AI can set reminders and recurring tasks for itself.
A background job (scheduler_run.py, every minute) runs them when they're due."""
import datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schedule as S

SAFE_GROUPS = {"web", "files", "toolbox", "agents", "schedule"}

def schedule_task(a):
    tools = [t.strip() for t in str(a.get("tools") or "web,files,toolbox").split(",") if t.strip()]
    blocked = [t for t in tools if t not in SAFE_GROUPS]
    tools = [t for t in tools if t in SAFE_GROUPS]
    t = S.add(a["prompt"], a["when"], a.get("name"), tools)
    nr = dt.datetime.fromisoformat(t["next_run"])
    msg = f"Scheduled '{t['name']}' (id {t['id']}): {S.describe(t['schedule'])}. Next run {nr:%a %d %b at %H:%M}."
    if blocked: msg += f" Left out {', '.join(blocked)}: scheduled tasks run unattended, so they can't control the Mac or run code."
    return msg + " The result will appear as a chat in Rivet with a notification."

def list_tasks(_):
    with S.Store() as st: tasks = list(st.tasks)
    if not tasks: return "No scheduled tasks."
    out = []
    for t in tasks:
        nr = f"next {dt.datetime.fromisoformat(t['next_run']):%a %d %b %H:%M}" if t.get("next_run") else "finished"
        out.append(f"{t['id']}  {t['name']}  [{S.describe(t['schedule'])}; {nr}; ran {t.get('runs', 0)}x]"
                   + (f"\n    last result: {t['last_result'][:120]}" if t.get("last_result") else ""))
    return "\n".join(out)

def cancel_task(a):
    with S.Store() as st:
        before = len(st.tasks); st.tasks = [t for t in st.tasks if t["id"] != a["id"]]; st.save()
    return "Cancelled." if len(st.tasks) < before else "No task with that id."

def run_task_now(a):
    with S.Store() as st:
        for t in st.tasks:
            if t["id"] == a["id"]:
                t["next_run"] = dt.datetime.now().isoformat(timespec="seconds"); st.save()
                return "It will run within a minute."
    return "No task with that id."

def O(req, **p): return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in p.items()}, "required": req}
TOOLS = {
 "schedule_task": (schedule_task, "Schedule a task or reminder for yourself to do later, once or repeating. The prompt runs unattended at that time and the answer is saved as a chat with a notification.",
                   O(["prompt", "when"], prompt="What to do then, written as an instruction to yourself", when="e.g. in 20 minutes, at 18:30, tomorrow 9am, daily 08:00, weekdays 9:00, every monday 9:00, every 2 hours",
                     name="short name", tools="comma list from: web, files, toolbox, agents (default web,files,toolbox)")),
 "list_tasks": (list_tasks, "List scheduled tasks with their next run and last result.", O([])),
 "cancel_task": (cancel_task, "Cancel a scheduled task by id.", O(["id"], id="task id from list_tasks")),
 "run_task_now": (run_task_now, "Run a scheduled task right away.", O(["id"], id="task id")),
}
def reply(i, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": i}; m["error" if error else "result"] = error or result
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
if __name__ == "__main__":
    for line in sys.stdin:
        try: m = json.loads(line)
        except Exception: continue
        i, method, p = m.get("id"), m.get("method"), m.get("params") or {}
        if method == "initialize": reply(i, {"protocolVersion": p.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}}, "serverInfo": {"name": "scheduler", "version": "1.0"}})
        elif method == "tools/list": reply(i, {"tools": [{"name": k, "description": d, "inputSchema": s} for k, (f, d, s) in TOOLS.items()]})
        elif method == "tools/call":
            try: text = str(TOOLS[p["name"]][0](p.get("arguments") or {}))
            except (ValueError, KeyError) as e: text = f"Couldn't schedule: {e}"
            except Exception as e: text = f"Error: {e}"
            reply(i, {"content": [{"type": "text", "text": text}]})
        elif i is not None: reply(i, error={"code": -32601, "message": "unknown method"})
