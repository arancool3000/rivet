"""Shared scheduling logic: parse 'when' phrases, store tasks, work out the next run."""
import datetime as dt, json, os, re, uuid, fcntl

FILE = os.path.expanduser("~/Rivet/schedule.json")
DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

def _time(s):
    """'8', '8am', '8:30', '18:30', '6.30pm', 'noon', 'midnight' -> (h, m)."""
    s = s.strip().lower().replace(".", ":")
    if s in ("noon", "midday"): return 12, 0
    if s == "midnight": return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m: raise ValueError(f"I don't understand the time '{s}'. Use something like 18:30 or 6:30pm.")
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12: h += 12
    if ap == "am" and h == 12: h = 0
    if not (0 <= h < 24 and 0 <= mi < 60): raise ValueError("That time doesn't exist.")
    return h, mi

def parse(when, now=None):
    """Turn a phrase into a schedule dict. Raises ValueError with a friendly message."""
    now = now or dt.datetime.now(); w = " ".join(when.lower().replace(",", " ").split())
    w = re.sub(r"^(on|at|every day at|everyday at)\s+", lambda m: "daily " if "day" in m.group(0) else "", w)
    if m := re.fullmatch(r"in\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)", w):
        n, u = float(m.group(1)), m.group(2)[0]
        delta = dt.timedelta(seconds=n) if u == "s" else dt.timedelta(minutes=n) if u == "m" else dt.timedelta(hours=n) if u == "h" else dt.timedelta(days=n)
        return {"kind": "once", "at": (now + delta).isoformat(timespec="seconds")}
    if m := re.fullmatch(r"every\s+(\d+)?\s*(m|min|mins|minutes?|h|hr|hrs|hours?)", w):
        n = int(m.group(1) or 1); mins = n * (60 if m.group(2)[0] == "h" else 1)
        if mins < 5: raise ValueError("The shortest repeat is every 5 minutes.")
        return {"kind": "interval", "minutes": mins}
    if m := re.fullmatch(r"(?:daily|every\s*day|each\s*day)(?:\s+at)?\s+(.+)", w):
        h, mi = _time(m.group(1)); return {"kind": "daily", "time": f"{h:02d}:{mi:02d}"}
    if m := re.fullmatch(r"(?:every\s+)?(weekdays?|weekends?)(?:\s+at)?\s+(.+)", w):
        h, mi = _time(m.group(2)); days = [0, 1, 2, 3, 4] if m.group(1).startswith("weekday") else [5, 6]
        return {"kind": "weekly", "days": days, "time": f"{h:02d}:{mi:02d}"}
    if m := re.fullmatch(r"(?:every|weekly|each)\s+([a-z]+(?:\s+(?:and\s+)?[a-z]+)*?)(?:\s+at)?\s+(\d[\d:.]*\s*(?:am|pm)?|noon|midnight)", w):
        days = [DAYS[d[:3]] for d in re.findall(r"[a-z]+", m.group(1)) if d[:3] in DAYS]
        if not days: raise ValueError("Name the day(s), e.g. 'every monday 9:00'.")
        h, mi = _time(m.group(2)); return {"kind": "weekly", "days": sorted(set(days)), "time": f"{h:02d}:{mi:02d}"}
    if m := re.fullmatch(r"(tomorrow|today)?\s*(?:at\s+)?(\d[\d:.]*\s*(?:am|pm)?|noon|midnight)\s*(tomorrow|today)?", w):
        h, mi = _time(m.group(2)); t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if "tomorrow" in (m.group(1) or "", m.group(3) or "") or t <= now: t += dt.timedelta(days=1)
        return {"kind": "once", "at": t.isoformat(timespec="seconds")}
    try:
        t = dt.datetime.fromisoformat(w.replace(" ", "T") if re.match(r"\d{4}-\d\d-\d\d \d", w) else w)
        if t <= now: raise ValueError("That date and time has already passed.")
        return {"kind": "once", "at": t.isoformat(timespec="seconds")}
    except ValueError as e:
        if "passed" in str(e): raise
    raise ValueError("I couldn't understand that schedule. Try: 'in 20 minutes', 'at 18:30', 'tomorrow 9am', "
                     "'daily 08:00', 'weekdays 9:00', 'every monday 9:00', 'every 2 hours', or '2026-12-01 09:00'.")

def next_run(s, after=None):
    after = after or dt.datetime.now()
    if s["kind"] == "once": t = dt.datetime.fromisoformat(s["at"]); return t if t > after - dt.timedelta(days=1) else None
    if s["kind"] == "interval": return after + dt.timedelta(minutes=s["minutes"])
    h, m = map(int, s["time"].split(":"))
    for add in range(0, 8):
        t = (after + dt.timedelta(days=add)).replace(hour=h, minute=m, second=0, microsecond=0)
        if t > after and (s["kind"] == "daily" or t.weekday() in s["days"]): return t
    return None

def describe(s):
    if s["kind"] == "once": return "once, " + dt.datetime.fromisoformat(s["at"]).strftime("%a %d %b %H:%M")
    if s["kind"] == "interval": return f"every {s['minutes']} min" if s["minutes"] < 60 else f"every {s['minutes'] / 60:g} h"
    if s["kind"] == "daily": return "daily at " + s["time"]
    names = [k.title() for k, v in DAYS.items() if v in s["days"]]
    return ("weekdays" if s["days"] == [0, 1, 2, 3, 4] else "weekends" if s["days"] == [5, 6] else ", ".join(names)) + " at " + s["time"]

class Store:
    """schedule.json with a file lock so the tool and the runner never clobber each other."""
    def __enter__(self):
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        self.f = open(FILE, "a+"); fcntl.flock(self.f, fcntl.LOCK_EX); self.f.seek(0)
        try: self.tasks = json.loads(self.f.read() or "[]")
        except ValueError: self.tasks = []
        return self
    def save(self):
        self.f.seek(0); self.f.truncate(); json.dump(self.tasks, self.f, indent=1); self.f.flush()
    def __exit__(self, *a): fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()

def add(prompt, when, name=None, tools=None):
    s = parse(when); nr = dt.datetime.fromisoformat(s["at"]) if s["kind"] == "once" else next_run(s)
    t = {"id": uuid.uuid4().hex[:8], "name": (name or prompt)[:60], "prompt": prompt, "schedule": s,
         "tools": tools if tools is not None else ["web", "files", "toolbox"], "next_run": nr.isoformat(timespec="seconds"),
         "created": dt.datetime.now().isoformat(timespec="seconds"), "last_run": None, "last_result": None, "runs": 0}
    with Store() as st: st.tasks.append(t); st.save()
    return t
