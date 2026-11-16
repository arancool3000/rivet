#!/usr/bin/env python3
"""Rivet connector: dials OUT to the workers.dev relay and bridges it to the local Rivet
server on 127.0.0.1:4480. No inbound port is opened on the home router.

Security:
  - Authenticates to the relay with CONNECT_SECRET (so only this Mac can be the backend).
  - Every forwarded request is tagged X-Rivet-Relay:1 and the phone's real IP; the local
    server treats those as untrusted and requires the 192-bit phone token on each one.
  - Only talks to 127.0.0.1:4480; never executes anything itself.
"""
import asyncio, base64, json, os, sys, urllib.request, urllib.parse, urllib.error
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 4480
LOCAL = f"http://127.0.0.1:{PORT}"

def cfg():
    try: return json.load(open(os.path.join(HERE, "config.json")))
    except Exception: return {}

def relay_ws_url(c):
    base = (c.get("relay") or "").rstrip("/")            # e.g. https://rivet-relay.<sub>.workers.dev
    if not base: return None
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/_connect"   # secret goes in a header, never the URL

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # The phone must receive 3xx redirects (and their Set-Cookie) itself — never follow them here,
    # or the pairing 302 would be swallowed and its auth cookie lost.
    def redirect_request(self, *a, **k): return None
_OPENER = urllib.request.build_opener(_NoRedirect)

# ---- forward one request to the local server, stream the response back over the socket ----
async def handle(ws, msg):
    rid = msg["id"]
    try:
        req = urllib.request.Request(LOCAL + msg["path"], method=msg["method"])
        hop = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade"}
        for k, v in (msg.get("headers") or {}).items():
            if k.lower() not in hop: req.add_header(k, v)
        data = base64.b64decode(msg["body"]) if msg.get("body") else None
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: _OPENER.open(req, data=data, timeout=120))
        hdrs = {k: v for k, v in resp.headers.items() if k.lower() not in hop}
        await ws.send(json.dumps({"t": "head", "id": rid, "status": resp.status, "headers": hdrs}))
        while True:
            chunk = await loop.run_in_executor(None, resp.read, 16384)
            if not chunk: break
            await ws.send(json.dumps({"t": "chunk", "id": rid, "d": base64.b64encode(chunk).decode()}))
        await ws.send(json.dumps({"t": "end", "id": rid}))
    except urllib.error.HTTPError as e:                  # non-2xx (incl. 302 pairing redirect, 403 bad token): pass it through verbatim with ALL headers
        try:
            hop = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade"}
            body = e.read()
            hdrs = {k: v for k, v in e.headers.items() if k.lower() not in hop}   # keep Set-Cookie, Location, etc.
            await ws.send(json.dumps({"t": "head", "id": rid, "status": e.code, "headers": hdrs}))
            if body: await ws.send(json.dumps({"t": "chunk", "id": rid, "d": base64.b64encode(body).decode()}))
            await ws.send(json.dumps({"t": "end", "id": rid}))
        except Exception:
            await ws.send(json.dumps({"t": "err", "id": rid}))
    except Exception:
        try: await ws.send(json.dumps({"t": "err", "id": rid}))
        except Exception: pass

_SEM = asyncio.Semaphore(12)                             # cap concurrent in-flight requests (DoS guard)
async def pump(ws):
    async def guarded(msg):
        async with _SEM: await handle(ws, msg)
    async for raw in ws:
        try: msg = json.loads(raw)
        except Exception: continue
        if msg.get("t") == "req" and isinstance(msg.get("path"), str) and msg["path"].startswith("/") \
           and msg.get("method") in ("GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"):
            asyncio.create_task(guarded(msg))            # validate the frame before acting on it

async def main():
    backoff = 2
    while True:
        c = cfg()
        url = relay_ws_url(c)
        secret = c.get("relay_secret") or ""
        if not (c.get("phone") and c.get("relay") and secret and url):
            await asyncio.sleep(5); continue
        try:
            async with websockets.connect(url, additional_headers={"x-connect-secret": secret},
                                          ping_interval=20, ping_timeout=20, max_size=12*1024*1024, open_timeout=20) as ws:
                backoff = 2
                sys.stderr.write("connector: linked to relay\n"); sys.stderr.flush()
                await pump(ws)
        except Exception as e:
            sys.stderr.write("connector: connection failed (" + type(e).__name__ + ")\n"); sys.stderr.flush()   # never echo the URL/secret
        await asyncio.sleep(backoff); backoff = min(backoff * 2, 30)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
