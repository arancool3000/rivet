"""Phase 1 (teacher loaded once): calibration data -> teacher targets -> per-layer sensitivity. Resumable."""
from common import *
from mlx_lm import stream_generate
from mlx_lm.utils import maybe_set_recommended_wired_limit
maybe_set_recommended_wired_limit()
os.makedirs(DATA, exist_ok=True)
model, tok = load(TEACHER)
log("teacher loaded")
# ---- A: calibration text. Chat samples written by the teacher itself + raw code/prose chunks.
TASKS = ["Explain {t} to a beginner, then give a worked example.", "Write a Python program related to {t}, with comments, then explain how it works.",
 "Write a short story involving {t}.", "Give a step-by-step guide to {t}, including common mistakes.", "Compare two approaches to {t} in a table, then recommend one.",
 "Write an email to a colleague about {t}.", "What are the three biggest misconceptions about {t}? Correct each.", "Write JavaScript code that demonstrates {t} and explain it line by line.",
 "Quiz me with five questions about {t}, then give the answers with reasoning.", "Summarise the history of {t} and where it is heading.",
 "Debug this idea: someone says {t} is simple. Argue both sides, then decide.", "Solve a tricky maths word problem about {t}, showing every step.",
 "Translate a short paragraph about {t} into French and Spanish, then explain the grammar.", "Write a bash script and a SQL query connected to {t}.",
 "hey, quick one, what do you reckon about {t}? keep it casual", "Write a poem about {t}, then analyse its structure.",
 "Design an electronics project using {t}: parts list, wiring, and firmware in C.", "Give me a detailed plan to learn {t} in 30 days.",
 "Explain the physics or mathematics behind {t} rigorously with formulas.", "Write a product description and FAQ for something based on {t}."]
TOPICS = ["neural networks", "the Raspberry Pi", "compound interest", "black holes", "3D printing", "the French Revolution", "sorting algorithms", "e-ink displays",
 "photosynthesis", "cryptography", "a rainy day in London", "chess openings", "stepper motors", "database indexing", "climate and weather", "the immune system",
 "probability and dice", "web servers", "ancient Rome", "music theory", "operating system memory", "cooking pasta", "electric cars", "regular expressions"]
gen_f = os.path.join(DATA, "chat.jsonl")
done = sum(1 for _ in open(gen_f)) if os.path.exists(gen_f) else 0
N_CHAT = 150
rng = np.random.default_rng(7); combos = [(int(a), int(b)) for a, b in zip(rng.integers(0, len(TASKS), 400), rng.integers(0, len(TOPICS), 400))]
combos = list(dict.fromkeys(combos))[:N_CHAT]
t0 = time.time()
with open(gen_f, "a") as out:
    for i, (a, b) in enumerate(combos):
        if i < done: continue
        p = tok.apply_chat_template([{"role": "user", "content": TASKS[a].format(t=TOPICS[b])}], add_generation_prompt=True, tokenize=False)
        txt = "".join(r.text for r in stream_generate(model, tok, p, max_tokens=380))
        out.write(json.dumps({"text": p + txt}) + "\n"); out.flush()
        if i % 10 == 0: log(f"chat sample {i+1}/{len(combos)}  ({(time.time()-t0)/max(1,i+1-done):.1f}s each)")
# raw chunks: code + docs already on this Mac (generic library files only, nothing personal)
import random, sysconfig, mlx_lm
SP = os.path.dirname(os.path.dirname(mlx_lm.__file__))
random.seed(3)
src = glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")) + glob.glob(os.path.join(SP, "*", "*.md")) \
    + glob.glob(os.path.join(SP, "*.dist-info", "METADATA"))
random.shuffle(src); raw = []
for f in src:
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    if len(t) > 6000:
        o = random.randrange(0, len(t) - 5000); raw.append(t[o:o + 5000])
    if len(raw) >= 220: break
# ---- B: teacher targets (top-1024 logits per position)
texts = [json.loads(l)["text"] for l in open(gen_f)] + raw
random.shuffle(texts)
n_valid = 24
for split, part in (("valid", texts[:n_valid]), ("train", texts[n_valid:])):
    os.makedirs(os.path.join(TGT, split), exist_ok=True)
    for i, t in enumerate(part):
        f = os.path.join(TGT, split, f"{i:05d}.safetensors")
        if os.path.exists(f): continue
        ids = tok.encode(t, add_special_tokens=False)[:640]
        if len(ids) < 64: continue
        x = mx.array(ids)
        lg = model(x[None])[0]
        idx = mx.argpartition(lg, kth=-1024, axis=-1)[..., -1024:]
        mx.save_safetensors(f, {"tokens": x, "idx": idx.astype(mx.int32), "logits": mx.take_along_axis(lg, idx, axis=-1).astype(mx.float16)})
        if i % 40 == 0: log(f"targets {split} {i+1}/{len(part)}")
log("targets done:", len(seqs("train")), "train,", len(seqs("valid")), "valid")
# ---- C: sensitivity of each layer's experts to fewer bits
sf = os.path.join(HERE, "sens.json")
sens = json.load(open(sf)) if os.path.exists(sf) else {}
V = seqs("valid")[:16]
if "base" not in sens: sens["base"] = kl_on(model, V); log("baseline (should be ~0):", sens["base"])
ex = experts(model)
for L in sorted(ex):
    for bits, gs in ((3, 64), (2, 64), (2, 32)):
        k = f"{L}:{bits}:{gs}"
        if k in sens: continue
        origs = {n: requant(m, bits, gs) for n, m in ex[L].items()}
        sens[k] = kl_on(model, V)
        for n, m in ex[L].items(): restore(m, origs[n])
        json.dump(sens, open(sf, "w")); log(f"layer {L} @ {bits}-bit/g{gs}: KL {sens[k][0]:.5f}  top1 {sens[k][1]:.4f}")
log("PHASE1 DONE")
