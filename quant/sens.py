from common import *
model, tok = load(TEACHER); V = seqs("valid")[:12]; ex = experts(model)
origs = {n: requant(m, 4, 64) for n, m in ex[1].items()}; r = kl_on(model, V[:6])
for n, m in ex[1].items(): restore(m, origs[n])
log(f"identity check L1 4b: KL {r[0]:.6f} top1 {r[1]:.4f}"); r0 = kl_on(model, V[:6]); log(f"after restore: KL {r0[0]:.6f}")
if r[0] > 0.003 or r0[0] > 1e-6: log("IDENTITY CHECK FAILED"); sys.exit(1)
sf = os.path.join(HERE, "sens.json"); sens = json.load(open(sf)) if os.path.exists(sf) else {}
for L in sorted(ex):
    for bits, gs in ((3, 64), (2, 64)):
        k = f"{L}:{bits}:{gs}"
        if k in sens: continue
        origs = {n: requant(m, bits, gs) for n, m in ex[L].items()}; sens[k] = kl_on(model, V)
        for n, m in ex[L].items(): restore(m, origs[n])
        json.dump(sens, open(sf, "w")); log(f"layer {L} @ {bits}-bit: KL {sens[k][0]:.5f} top1 {sens[k][1]:.4f}")
log("SENS DONE")
