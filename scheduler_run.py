#!/usr/bin/env python3
"""Runs due scheduled tasks (launchd calls this every minute). Each task goes through Rivet's
server unattended: no approvals, so nothing that clicks, controls apps or runs code."""
import datetime as dt, json, os, subprocess, sys, time, urllib.request, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schedule as S

H = os.path.expanduser("~"); URL = "http://127.0.0.1:4480"
PY = f"{H}/LocalVoice/.venv/bin/python"

def up():
    try: urllib.request.urlopen(URL + "/state", timeout=3); return True
    except Exception: return False

def post(path, body, timeout=60):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json", "X-Rivet": "1"})
    return urllib.request.urlopen(req, timeout=timeout)

def notify(title, msg):
    q = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{q(msg[:180])}" with title "{q(title[:60])}" sound name "Glass"'], capture_output=True)

def run(task):
    started = False
    if not up():
        subprocess.Popen([PY, f"{H}/Rivet/server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "RIVET_HEADLESS": "1"})
        started = True
        for _ in range(40):
            if up(): break
            time.sleep(0.5)
    cid = uuid.uuid4().hex
    prompt = (f"[Scheduled task '{task['name']}', running automatically at {dt.datetime.now():%A %d %B %H:%M}. "
              f"Nobody is watching, so finish the job and reply with a short, useful result.]\n\n{task['prompt']}")
    body = {"messages": [{"role": "user", "content": prompt}], "model": "auto", "think": "low", "tools": task.get("tools", []),
            "chat_id": cid, "unattended": True}
    text, tools_used, err = "", [], None
    try:
        for line in post("/chat", body, timeout=1800):
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            e = json.loads(line[5:])
            if "text_replace" in e: text = e["text_replace"] or text
            if e.get("tool"): tools_used.append({"name": e["tool"], "args": e.get("args"), "id": e.get("id"), "result": None})
            if e.get("tool_result") is not None and tools_used: tools_used[-1]["result"] = e["tool_result"]
            if e.get("error"): err = e["error"]
            if e.get("done"): label = e.get("label")
    except Exception as e: err = str(e)
    result = err or text or "(no answer)"
    chat = {"id": cid, "title": f"Scheduled: {task['name']}"[:60], "side": [], "messages": [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": text, "error": err, "tools": tools_used, "label": "Scheduled task"}]}
    try: post("/save", chat)
    except Exception: pass
    notify(f"Rivet: {task['name']}", ("Problem: " if err else "") + (result.replace("\n", " ")))
    return result

def models_busy():
    """True if LM Studio is in the middle of a reply. Two requests at once on one model can crash its engine."""
    try:
        out = subprocess.run([os.path.expanduser("~/.lmstudio/bin/lms"), "ps", "--json"], capture_output=True, text=True, timeout=20).stdout
        return any((m.get("status") or "idle").lower() not in ("idle", "loaded", "ready") for m in json.loads(out or "[]"))
    except Exception: return False

def main():
    now = dt.datetime.now(); due = []
    with S.Store() as st:
        if not any(t.get("next_run") and dt.datetime.fromisoformat(t["next_run"]) <= now for t in st.tasks): return
    if models_busy(): return            # try again next minute
    with S.Store() as st:
        for t in st.tasks:
            if t.get("next_run") and dt.datetime.fromisoformat(t["next_run"]) <= now:
                due.append(dict(t))
                nr = None if t["schedule"]["kind"] == "once" else S.next_run(t["schedule"], now)
                t["next_run"] = nr.isoformat(timespec="seconds") if nr else None
        st.save()
    for t in due:
        res = run(t)
        with S.Store() as st:
            for x in st.tasks:
                if x["id"] == t["id"]:
                    x["last_run"] = dt.datetime.now().isoformat(timespec="seconds"); x["last_result"] = res[:2000]; x["runs"] = x.get("runs", 0) + 1
            st.tasks = [x for x in st.tasks if x.get("next_run") or x["schedule"]["kind"] != "once" or (dt.datetime.now() - dt.datetime.fromisoformat(x.get("last_run") or x["created"])).days < 7]
            st.save()

if __name__ == "__main__":
    main()
