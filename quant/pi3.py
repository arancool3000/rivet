"""pi3.py — same rebuild for Pi 3 (gpt-oss 20B): experts mxfp4 (4.25 bpw) -> affine 3-bit (3.5 bpw), distilled against
the original's own outputs. Stages are resumable. Output: out/P3 (loadable folder)."""
from common import *
import random, sysconfig, shutil, mlx_lm
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import stream_generate
from mlx_lm.utils import maybe_set_recommended_wired_limit
maybe_set_recommended_wired_limit()
T3 = os.path.expanduser("~/.lmstudio/models/aran/2-Forge-gpt-oss-20b")
D3, G3, OUT = os.path.join(HERE, "data3"), os.path.join(HERE, "targets3"), os.path.join(HERE, "out", "P3")
for d in (D3, OUT): os.makedirs(d, exist_ok=True)
STEPS, NL = int(sys.argv[1]) if len(sys.argv) > 1 else 75, 24
model, tok = load(T3); log("P3 teacher loaded")
# mlx has no gradient for attention with sinks, so training uses a hand-written (checked) equivalent
import mlx_lm.models.gpt_oss as G
_fast = G.scaled_dot_product_attention
def _slow(q, k, v, cache, scale, mask=None, sinks=None):
    B, H, L, D = q.shape; r = H // k.shape[1]
    if r > 1: k, v = mx.repeat(k, r, axis=1), mx.repeat(v, r, axis=1)
    sc = ((q * scale) @ k.swapaxes(-1, -2)).astype(mx.float32)
    if isinstance(mask, str): mask = mx.triu(mx.full((L, k.shape[2]), -mx.inf), k=1)
    if mask is not None: sc = mx.where(mask, sc, -mx.inf) if mask.dtype == mx.bool_ else sc + mask.astype(mx.float32)
    if sinks is not None: sc = mx.concatenate([sc, mx.broadcast_to(sinks.astype(mx.float32).reshape(1, H, 1, 1), (B, H, L, 1))], -1)
    w = mx.softmax(sc, -1)
    if sinks is not None: w = w[..., :-1]
    return w.astype(v.dtype) @ v
_x = mx.array(tok.encode("The quick brown fox jumps over the lazy dog. " * 12, add_special_tokens=False))[None]
_a = model(_x)[0].astype(mx.float32); G.scaled_dot_product_attention = _slow; _b = model(_x)[0].astype(mx.float32)
_agree = (mx.argmax(_a, -1) == mx.argmax(_b, -1)).mean().item(); _d = mx.abs(_a - _b).max().item()
log(f"P3 attention check: top1 agreement {_agree:.4f}, max logit diff {_d:.3f}")
assert _agree > 0.98, "hand-written attention does not match"
G.scaled_dot_product_attention = _fast
# ---- 1: chat samples written by the teacher
import importlib.util
src_txt = open(os.path.join(HERE, "phase1.py")).read()
ns = {}; exec(src_txt[src_txt.index("TASKS = ["):src_txt.index("gen_f = ")], ns); TASKS, TOPICS = ns["TASKS"], ns["TOPICS"]
gen_f = os.path.join(D3, "chat.jsonl"); done = sum(1 for _ in open(gen_f)) if os.path.exists(gen_f) else 0
rng = np.random.default_rng(7); combos = [(int(a), int(b)) for a, b in zip(rng.integers(0, len(TASKS), 400), rng.integers(0, len(TOPICS), 400))]
combos = list(dict.fromkeys(combos))[:150]; t0 = time.time()
with open(gen_f, "a") as out:
    for i, (a, b) in enumerate(combos):
        if i < done: continue
        p = tok.apply_chat_template([{"role": "user", "content": TASKS[a].format(t=TOPICS[b])}], add_generation_prompt=True, tokenize=False)
        ids = tok.encode(p, add_special_tokens=False)
        for r in stream_generate(model, tok, p, max_tokens=380): ids.append(r.token)
        out.write(json.dumps({"ids": [int(x) for x in ids]}) + "\n"); out.flush()
        if i % 10 == 0: log(f"P3 chat sample {i+1}/{len(combos)}  ({(time.time()-t0)/max(1,i+1-done):.1f}s each)")
# ---- 2: teacher targets
SP = os.path.dirname(os.path.dirname(mlx_lm.__file__)); random.seed(3)
src = glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")) + glob.glob(os.path.join(SP, "*", "*.md")) + glob.glob(os.path.join(SP, "*.dist-info", "METADATA"))
random.shuffle(src); raw = []
for f in src:
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    if len(t) > 6000:
        o = random.randrange(0, len(t) - 5000); raw.append(tok.encode(t[o:o + 5000], add_special_tokens=False))
    if len(raw) >= 220: break
seqs_ = [json.loads(l)["ids"] for l in open(gen_f)] + raw; random.shuffle(seqs_)
for split, part in (("valid", seqs_[:24]), ("train", seqs_[24:])):
    os.makedirs(os.path.join(G3, split), exist_ok=True)
    for i, ids in enumerate(part):
        f = os.path.join(G3, split, f"{i:05d}.safetensors")
        if os.path.exists(f) or len(ids) < 64: continue
        x = mx.array(ids[:640]); lg = model(x[None])[0]
        idx = mx.argpartition(lg, kth=-1024, axis=-1)[..., -1024:]
        mx.save_safetensors(f, {"tokens": x, "idx": idx.astype(mx.int32), "logits": mx.take_along_axis(lg, idx, axis=-1).astype(mx.float16)})
V, T = sorted(glob.glob(os.path.join(G3, "valid", "*"))), sorted(glob.glob(os.path.join(G3, "train", "*")))
log("P3 targets:", len(T), "train,", len(V), "valid")
st = os.path.join(OUT, "state.json"); ck = os.path.join(OUT, "ckpt.safetensors")
state = json.load(open(st)) if os.path.exists(st) else {"block": 0}
if "base" not in state: state["base"] = kl_on(model, V); log(f"P3 identity check (should be ~0, 1.0): KL {state['base'][0]:.5f} top1 {state['base'][1]:.4f}")
# ---- 3: re-quantise every expert projection to affine 3-bit
ex = experts(model); assert len(ex) == NL, len(ex)
for L in sorted(ex):
    for m in ex[L].values():
        w = mx.dequantize(m.weight, m.scales, None, group_size=m.group_size, bits=m.bits, mode=m.mode).astype(mx.float32)
        qw, qs, qb = mx.quantize(w, group_size=64, bits=3)
        m.weight, m.scales, m.biases = qw, qs.astype(mx.bfloat16), qb.astype(mx.bfloat16)
        m.bits, m.group_size, m.mode = 3, 64, "affine"; mx.eval(m.weight, m.scales, m.biases)
    mx.clear_cache()
if os.path.exists(ck):
    for k, v in mx.load(ck).items():
        L, n, kk = k.split("."); setattr(ex[int(L)][n], kk, v)
    log("P3 resumed at block", state["block"])
if "kl0" not in state: state["kl0"] = kl_on(model, V); log(f"P3 before tuning: KL {state['kl0'][0]:.4f} top1 {state['kl0'][1]:.4f}"); json.dump(state, open(st, "w"))
# ---- 4: distil, 4 blocks of 6 layers
def loss_fn(params, d):
    G.scaled_dot_product_attention = _slow
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), params))
    n = int(os.environ.get("SEQ", "224")); lg = model(d["tokens"][:n][None])[0]
    s = mx.take_along_axis(lg, d["idx"][:n], axis=-1).astype(mx.float32); t = d["logits"][:n].astype(mx.float32)
    lt, ls = t - mx.logsumexp(t, -1, keepdims=True), s - mx.logsumexp(s, -1, keepdims=True)
    return (mx.exp(lt) * (lt - ls)).sum(-1).mean()
BL = int(os.environ.get("BL", "3")); blocks = [list(range(i, i + BL)) for i in range(0, NL, BL)]; rng = np.random.default_rng(1)
for bi, block in enumerate(blocks):
    if bi < state["block"] or STEPS == 0: continue
    model.freeze()
    for L in block:
        for m in ex[L].values(): m.unfreeze(keys=["scales", "biases"], recurse=False)
    params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())
    opt = optim.Adam(learning_rate=float(os.environ.get("LR", "2e-6"))); vg = mx.value_and_grad(loss_fn); run = 0.0; t0 = time.time()
    for it in range(STEPS):
        d = mx.load(T[int(rng.integers(len(T)))])
        loss, grads = vg(params, d); params = opt.apply_gradients(grads, params); mx.eval(loss, params); run += loss.item(); mx.clear_cache()
        if (it + 1) % 25 == 0:
            log(f"P3 block {bi+1}/{len(blocks)} step {it+1}/{STEPS} loss {run/25:.4f}  {(time.time()-t0)/(it+1):.1f}s/step peak {mx.get_peak_memory()/1e9:.1f}GB"); run = 0.0
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), params)); del params, opt, grads; mx.clear_cache()
    mx.save_safetensors(ck, {f"{L}.{n}.{k}": getattr(m, k) for L in ex for n, m in ex[L].items() for k in ("scales", "biases")})
    state["block"] = bi + 1; json.dump(state, open(st, "w"))
G.scaled_dot_product_attention = _fast
model.freeze(); state["kl1"] = kl_on(model, V); json.dump(state, open(st, "w"))
log(f"P3 RESULT 3-bit: KL {state['kl0'][0]:.4f} -> {state['kl1'][0]:.4f}   top1 {state['kl0'][1]:.4f} -> {state['kl1'][1]:.4f}")
# ---- 5: save a loadable folder
new = {f"model.layers.{L}.mlp.experts.{n}.{k}": getattr(m, k) for L in ex for n, m in ex[L].items() for k in ("weight", "scales", "biases")}
mx.eval(list(new.values())); del model; mx.clear_cache(); hit = 0
for f in sorted(os.listdir(T3)):
    s = os.path.join(T3, f)
    if f.endswith(".safetensors"):
        d = dict(mx.load(s))
        for k in list(d):
            if k in new: d[k] = new.pop(k); hit += 1
        if f.startswith("model-00003"): d.update(new); hit += len(new); new = {}   # the new .biases tensors
        mx.save_safetensors(os.path.join(OUT, f), d, metadata={"format": "mlx"}); del d; mx.clear_cache()
    elif os.path.isfile(s) and f not in ("config.json", "model.safetensors.index.json"): shutil.copy(s, OUT)
assert hit == NL * 3 * 3, hit
c = json.load(open(os.path.join(T3, "config.json")))
for L in range(NL):
    for n in ("gate_proj", "up_proj", "down_proj"):
        for key in ("quantization", "quantization_config"): c[key][f"model.layers.{L}.mlp.experts.{n}"] = {"group_size": 64, "bits": 3, "mode": "affine"}
json.dump(c, open(os.path.join(OUT, "config.json"), "w"), indent=2)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("model-")) / 1e9
log(f"P3 SAVED {OUT}  {sz:.2f} GB"); log("P3 TUNE DONE")
