from common import *
model, tok = load(TEACHER); V = seqs("valid")[:8]; ex = experts(model)
for L in (1, 20):
    for bits, gs in ((4, 64), (6, 64), (3, 64)):
        origs = {n: requant(m, bits, gs) for n, m in ex[L].items()}; r = kl_on(model, V)
        for n, m in ex[L].items(): restore(m, origs[n])
        log(f"diag L{L} all {bits}b: KL {r[0]:.5f} top1 {r[1]:.4f}")
    for n, m in ex[L].items():
        o = requant(m, 3, 64); r = kl_on(model, V); restore(m, o); log(f"diag L{L} only {n} 3b: KL {r[0]:.5f}")
del model; mx.clear_cache()
m4, _ = load(os.path.expanduser("~/.lmstudio/models/aran/4-Pi4-Gemma4-26B-3bit"))
r = kl_on(m4, seqs("valid")); log(f"PI4 vs PI5 on all valid: KL {r[0]:.5f} top1 {r[1]:.4f}"); log("DIAG DONE")
