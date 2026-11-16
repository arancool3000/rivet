#!/usr/bin/env python3
"""Keep MLX model weights pinned in RAM while a model is loaded, so macOS never pushes them to swap (SSD wear).
Measured 16 Nov 2026: the wired limit alone only holds during generation; at idle the pin drops within 10 s and the
weights start going to swap. A tiny GPU task every second keeps them wired (3 s only held 65% of the time). It pauses while the engine is generating
or loading, so it never runs alongside real work. Also reads prompts in 512-token chunks (lower peak RAM).
Idempotent; re-run after LM Studio engine updates (Rivet runs it at start)."""
import glob, os
MARK = "# --- rivet-pin v5 ---"
PATCH = '''

# --- rivet-pin v5 ---
import threading as _rv_threading, time as _rv_time
_rivet_lock, _rivet_busy, _rivet_started = _rv_threading.Lock(), [0], [False]
def _rivet_keep_wired():
    import mlx.core as _mx
    while True:
        _rv_time.sleep(1)
        try:
            with _rivet_lock:
                if _rivet_busy[0] <= 0: _mx.eval(_mx.array(1.0) + 1)
        except Exception:
            pass
class _RivetIter:
    """Marks the engine busy for as long as a reply is being produced."""
    def __init__(self, it): self._it, self._open = it, True
    def __iter__(self): return self
    def _done(self):
        if self._open:
            self._open = False
            with _rivet_lock: _rivet_busy[0] -= 1
    def __next__(self):
        try: return next(self._it)
        except BaseException: self._done(); raise
    def close(self):
        try:
            c = getattr(self._it, "close", None)
            if c: c()
        finally: self._done()
    def __del__(self): self._done()
    def __getattr__(self, n): return getattr(self._it, n)
_rivet_orig_create_generator = create_generator
def create_generator(*a, **k):
    with _rivet_lock: _rivet_busy[0] += 1
    try: it = _rivet_orig_create_generator(*a, **k)
    except BaseException:
        with _rivet_lock: _rivet_busy[0] -= 1
        raise
    return _RivetIter(it)
_rivet_orig_load_model = load_model
def load_model(*a, **k):
    with _rivet_lock: _rivet_busy[0] += 1
    try:
        # read long prompts in 512-token chunks instead of the default: ~0.45 GB lower peak RAM, same speed, same output
        if k.get("prefill_step_size") is None:
            try:
                kit = _rivet_orig_load_model(*a, **dict(k, prefill_step_size=512))
            except TypeError:
                kit = _rivet_orig_load_model(*a, **k)
        else:
            kit = _rivet_orig_load_model(*a, **k)
        try:
            if not k.get("vocab_only"):
                import mlx.core as _mx
                _rec = int(_mx.device_info()["max_recommended_working_set_size"])
                if not getattr(_mx, "_rivet_pinned", False):
                    _real = _mx.set_wired_limit
                    def _keep(limit, _real=_real, _rec=_rec):      # the engine drops the limit to 0 when idle; never let it
                        return _real(max(int(limit), _rec))
                    _mx.set_wired_limit = _keep; _mx._rivet_pinned = True
                _mx.set_wired_limit(_rec)
                if not _rivet_started[0]:
                    _rivet_started[0] = True
                    _rv_threading.Thread(target=_rivet_keep_wired, daemon=True).start()
        except Exception:
            pass
        return kit
    finally:
        with _rivet_lock: _rivet_busy[0] -= 1
'''
n = 0
for f in glob.glob(os.path.expanduser("~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-*/lib/python*/site-packages/mlx_engine/generate.py")):
    s = open(f).read()
    if MARK in s: continue
    if os.path.exists(f + ".orig"): s = open(f + ".orig").read()
    else: open(f + ".orig", "w").write(s)
    open(f, "w").write(s + PATCH); n += 1
print("patched", n)
