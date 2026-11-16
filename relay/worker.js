// Rivet relay — a public workers.dev front door for the Mac's local Rivet server.
// The Mac dials OUT to this Worker (so no inbound port on the home router). The phone
// talks to this Worker; every request is forwarded over the Mac's socket and streamed back.
//
// Trust model:
//  - Only the Mac, holding CONNECT_SECRET, may register as the backend (/_connect).
//  - Every phone request is tagged x-rivet-relay:1 and the real client IP, and the Mac
//    re-validates the 192-bit phone token on each request. This Worker never sees or stores it.
//  - Per-IP lockout here blunts brute force before it reaches the Mac.

const te = new TextEncoder();
function safeEq(a, b) {                       // constant-time string compare
  a = te.encode(String(a || "")); b = te.encode(String(b || ""));
  if (a.length !== b.length) return false;
  let r = 0; for (let i = 0; i < a.length; i++) r |= a[i] ^ b[i];
  return r === 0;
}
const b64 = (buf) => { let s = ""; const u = new Uint8Array(buf); for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s); };
const MAX_BODY = 8 * 1024 * 1024;                 // 8 MB cap on request bodies (DoS guard)
const SEC = { "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store" };
function secured(resp) { for (const [k, v] of Object.entries(SEC)) resp.headers.set(k, v); return resp; }   // security headers on every response, incl. errors
const unb64 = (s) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };

export default {
  async fetch(req, env) {
    const id = env.RELAY.idFromName("singleton");
    return env.RELAY.get(id).fetch(req);
  },
};

export class Relay {
  constructor(state, env) {
    this.state = state; this.env = env;
    this.mac = null;                 // the Mac connector WebSocket
    this.pending = new Map();        // request id -> { controller, headResolve }
    this.nextId = 1;
    this.fails = new Map();          // ip -> { n, until }  (brute-force lockout)
  }

  locked(ip) {
    const f = this.fails.get(ip);
    return f && f.until > Date.now();
  }
  noteFail(ip) {
    const f = this.fails.get(ip) || { n: 0, until: 0 };
    f.n++; if (f.n >= 10) f.until = Date.now() + 10 * 60 * 1000;   // 10 strikes -> 10 min lockout
    this.fails.set(ip, f);
  }

  onMacMessage(raw) {
    let m; try { m = JSON.parse(raw); } catch { return; }
    const p = this.pending.get(m.id); if (!p) return;
    if (m.t === "head") { p.headResolve({ status: m.status, headers: m.headers || {} }); }
    else if (m.t === "chunk") { try { p.controller.enqueue(unb64(m.d)); } catch {} }
    else if (m.t === "end") { try { p.controller.close(); } catch {} this.pending.delete(m.id); }
    else if (m.t === "err") {
      if (p.headResolve) p.headResolve({ status: 502, headers: { "content-type": "text/plain" }, errText: "Mac error" });
      try { p.controller.close(); } catch {} this.pending.delete(m.id);
    }
  }

  async fetch(req) {
    const url = new URL(req.url);

    // --- Mac connector registering as the backend ---
    if (url.pathname === "/_connect") {
      const cip = req.headers.get("cf-connecting-ip") || "0.0.0.0";
      if (this.locked(cip)) return secured(new Response("Too many attempts.", { status: 429 }));
      // Secret comes in a header (not the URL) so it never lands in request logs. Also accept the legacy ?s= during rollout.
      const presented = req.headers.get("x-connect-secret") || url.searchParams.get("s") || "";
      if (!safeEq(presented, this.env.CONNECT_SECRET)) { this.noteFail(cip); return secured(new Response("forbidden", { status: 403 })); }
      this.fails.delete(cip);
      // If a healthy backend is already connected, REFUSE the newcomer rather than letting it hijack all phone traffic.
      if (this.mac && this.mac.readyState === 1) return secured(new Response("backend already connected", { status: 409 }));
      const [client, server] = Object.values(new WebSocketPair());
      server.accept();
      if (this.mac) { try { this.mac.close(1000, "replaced"); } catch {} }
      for (const p of this.pending.values()) { try { p.controller.close(); } catch {} }   // fail in-flight requests fast instead of leaking
      this.pending.clear();
      this.mac = server;
      server.addEventListener("message", (e) => this.onMacMessage(e.data));
      const drop = () => { if (this.mac === server) this.mac = null; for (const p of this.pending.values()) { try { p.controller.close(); } catch {} } this.pending.clear(); };
      server.addEventListener("close", drop); server.addEventListener("error", drop);
      return new Response(null, { status: 101, webSocket: client });
    }

    // --- phone request -> forward to the Mac ---
    const ip = req.headers.get("cf-connecting-ip") || "0.0.0.0";
    if (this.locked(ip)) return secured(new Response("Too many attempts. Try again later.", { status: 429 }));
    if (!this.mac) return secured(new Response("Rivet is offline. Open Rivet on your Mac and turn on Phone access.", { status: 503 }));
    // Reject oversized bodies BEFORE buffering anything (unauthenticated DoS guard).
    if (+(req.headers.get("content-length") || 0) > MAX_BODY) return secured(new Response("Request too large.", { status: 413 }));

    const id = this.nextId++;
    const hdrs = {}; for (const [k, v] of req.headers) hdrs[k.toLowerCase()] = v;
    delete hdrs["x-rivet-relay"]; delete hdrs["x-real-ip"];    // only we may set these
    hdrs["x-rivet-relay"] = "1"; hdrs["x-real-ip"] = ip;
    let bodyBuf = null;
    if (req.method !== "GET" && req.method !== "HEAD") {
      bodyBuf = await req.arrayBuffer();
      if (bodyBuf.byteLength > MAX_BODY) return secured(new Response("Request too large.", { status: 413 }));
    }

    let controller;
    const stream = new ReadableStream({ cancel: () => { try { this.pending.delete(id); } catch {} }, start(c) { controller = c; } });
    const headP = new Promise((resolve) => this.pending.set(id, { controller: null, headResolve: resolve }));
    const slot = this.pending.get(id); slot.controller = controller;

    try {
      this.mac.send(JSON.stringify({ t: "req", id, method: req.method, path: url.pathname + url.search, headers: hdrs, body: bodyBuf ? b64(bodyBuf) : null }));
    } catch {
      this.pending.delete(id); return secured(new Response("Mac connection dropped.", { status: 502 }));
    }
    const timeout = setTimeout(() => { const p = this.pending.get(id); if (p && p.headResolve) p.headResolve({ status: 504, headers: {}, errText: "Mac timed out" }); }, 30000);
    const head = await headP; clearTimeout(timeout);

    // track auth failures the Mac reports (403 on a bad/absent token); only a genuine success clears the counter
    if (head.status === 403) this.noteFail(ip); else if (head.status < 400) this.fails.delete(ip);
    if (head.errText) { try { controller.close(); } catch {} this.pending.delete(id); }

    return secured(new Response(head.errText ? head.errText : stream, { status: head.status, headers: new Headers(head.headers || {}) }));
  }
}
