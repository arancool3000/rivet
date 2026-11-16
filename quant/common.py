import os, sys, json, time, glob, re, numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER = os.path.expanduser("~/.lmstudio/models/aran/1-Apex-Gemma4-26B")
DATA, TGT = os.path.join(HERE, "data"), os.path.join(HERE, "targets")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(os.path.join(HERE, "progress.log"), "a").write(s + "\n")
def experts(model):
    """{layer_index: {proj_name: module}} for every expert projection."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantizedSwitchLinear):
            L = int(re.search(r"layers\.(\d+)\.", name).group(1))
            out.setdefault(L, {})[name.split(".")[-1]] = m
    return out
def requant(m, bits, gs, src=None):
    """Re-quantise an expert projection in place from its (4-bit) source tensors. Returns the originals."""
    orig = src or (m.weight, m.scales, m.biases, m.bits, m.group_size)
    # float32 throughout: in bf16 the dequantised weights lose the low bits and the new grid comes out wrong
    w = mx.dequantize(orig[0], orig[1].astype(mx.float32), orig[2].astype(mx.float32), group_size=orig[4], bits=orig[3])
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=bits)
    m.weight, m.scales, m.biases = qw, qs.astype(orig[1].dtype), qb.astype(orig[2].dtype)
    m.bits, m.group_size = bits, gs
    mx.eval(m.weight, m.scales, m.biases)
    return orig
def restore(m, orig):
    m.weight, m.scales, m.biases, m.bits, m.group_size = orig
def seqs(split):
    return sorted(glob.glob(os.path.join(TGT, split, "*.safetensors")))
def kl_on(model, files):
    """Mean KL(teacher || model) per token and top-1 agreement, over cached teacher targets."""
    tot = agree = n = 0
    for f in files:
        d = mx.load(f)
        lg = model(d["tokens"][None])[0]
        s = mx.take_along_axis(lg, d["idx"], axis=-1).astype(mx.float32)
        t = d["logits"].astype(mx.float32)
        lt, ls = t - mx.logsumexp(t, -1, keepdims=True), s - mx.logsumexp(s, -1, keepdims=True)
        kl = (mx.exp(lt) * (lt - ls)).sum(-1)
        top_t = mx.take_along_axis(d["idx"], mx.argmax(t, -1, keepdims=True), -1)[:, 0]
        a = (mx.argmax(lg, -1) == top_t).sum()
        mx.eval(kl, a); tot += kl.sum().item(); agree += a.item(); n += kl.shape[0]
    return tot / n, agree / n
