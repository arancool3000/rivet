"""tune.py NAME PLAN STEPS   PLAN = bits per layer, e.g. "3*12,2*18".  Re-quantise experts, distil against Pi 5's
cached outputs (block by block to fit in RAM), evaluate, save a loadable model folder. Resumable per block."""
from common import *
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from mlx_lm.utils import maybe_set_recommended_wired_limit
maybe_set_recommended_wired_limit()
name, plan, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
bits = [int(b) for part in plan.split(",") for b in [part.split("*")[0]] * int(part.split("*")[1])]
assert len(bits) == 30
OUT = os.path.join(HERE, "out", name); os.makedirs(OUT, exist_ok=True)
model, tok = load(TEACHER); ex = experts(model)
for L in sorted(ex):
    if bits[L] != 4:
        for m in ex[L].values(): requant(m, bits[L], 64)
mx.clear_cache()
V, T = seqs("valid"), seqs("train")
ck = os.path.join(OUT, "ckpt.safetensors"); st = os.path.join(OUT, "state.json")
state = json.load(open(st)) if os.path.exists(st) else {"block": 0}
def expert_params():
    return {f"{L}.{n}.{k}": getattr(m, k) for L in ex if bits[L] != 4 for n, m in ex[L].items() for k in ("scales", "biases")}
if os.path.exists(ck):
    for k, v in mx.load(ck).items():
        L, n, kk = k.split("."); setattr(ex[int(L)][n], kk, v)
    log(name, "resumed at block", state["block"])
if "kl0" not in state: state["kl0"] = kl_on(model, V); log(name, f"before tuning: KL {state['kl0'][0]:.4f} top1 {state['kl0'][1]:.4f}")
def loss_fn(params, d):
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), params))
    lg = model(d["tokens"][None])[0]
    s = mx.take_along_axis(lg, d["idx"], axis=-1).astype(mx.float32); t = d["logits"].astype(mx.float32)
    lt, ls = t - mx.logsumexp(t, -1, keepdims=True), s - mx.logsumexp(s, -1, keepdims=True)
    return (mx.exp(lt) * (lt - ls)).sum(-1).mean()
blocks = [list(range(i, i + 6)) for i in range(0, 30, 6)]
rng = np.random.default_rng(1)
for bi, block in enumerate(blocks):
    if bi < state["block"] or steps == 0: continue
    model.freeze(); n_tr = 0
    for L in block:
        if bits[L] == 4: continue
        for m in ex[L].values(): m.unfreeze(keys=["scales", "biases"], recurse=False); n_tr += 1
    if not n_tr: continue
    params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())
    opt = optim.Adam(learning_rate=float(os.environ.get("LR", "2e-6")))
    vg = mx.value_and_grad(loss_fn); run = 0.0; t0 = time.time()
    for it in range(steps):
        d = mx.load(T[int(rng.integers(len(T)))])
        loss, grads = vg(params, d)
        params = opt.apply_gradients(grads, params); mx.eval(loss, params); run += loss.item()
        if (it + 1) % 20 == 0:
            log(name, f"block {bi+1}/{len(blocks)} step {it+1}/{steps} loss {run/20:.4f}  {(time.time()-t0)/(it+1):.1f}s/step peak {mx.get_peak_memory()/1e9:.1f}GB"); run = 0.0
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), params)); del params, opt, grads; mx.clear_cache()
    mx.save_safetensors(ck, expert_params()); state["block"] = bi + 1; json.dump(state, open(st, "w"))
model.freeze()
state["kl1"] = kl_on(model, V); json.dump(state, open(st, "w"))
log(name, f"RESULT plan {plan}: KL {state['kl0'][0]:.4f} -> {state['kl1'][0]:.4f}   top1 {state['kl0'][1]:.4f} -> {state['kl1'][1]:.4f}")
# ---- save as a loadable model folder (same layout as the original, experts replaced)
new = {}
for L in ex:
    if bits[L] == 4: continue
    for n, m in ex[L].items():
        for k in ("weight", "scales", "biases"): new[f"layers.{L}.experts.switch_glu.{n}.{k}"] = getattr(m, k)
mx.eval(list(new.values())); del model; mx.clear_cache()
import shutil
for f in sorted(os.listdir(TEACHER)):
    src = os.path.join(TEACHER, f)
    if f.endswith(".safetensors"):
        d = dict(mx.load(src))
        for k in list(d):
            suf = k[k.index("layers."):] if "experts.switch_glu" in k else None
            if suf in new: d[k] = new[suf]
        mx.save_safetensors(os.path.join(OUT, f), d, metadata={"format": "mlx"}); del d; mx.clear_cache()
    elif os.path.isfile(src) and f != "config.json": shutil.copy(src, OUT)
c = json.load(open(os.path.join(TEACHER, "config.json")))
for L in range(30):
    if bits[L] == 4: continue
    for n in ("gate_proj", "up_proj", "down_proj"):
        for key in ("quantization", "quantization_config"):
            c[key][f"language_model.model.layers.{L}.experts.switch_glu.{n}"] = {"group_size": 64, "bits": bits[L]}
json.dump(c, open(os.path.join(OUT, "config.json"), "w"), indent=2)
assert sum(1 for L in range(30) if bits[L] != 4) == 0 or any(True for _ in new), "nothing replaced"
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("model-")) / 1e9
log(name, f"SAVED {OUT}  {sz:.2f} GB"); log(name, "TUNE DONE")
