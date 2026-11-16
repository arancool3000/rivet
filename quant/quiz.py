"""quiz.py MODEL_DIR LABEL — 100 exact-answer questions (maths, logic, strings, code tracing). Same questions for every model."""
from common import *
from mlx_lm import generate
import random
random.seed(11); Q = []
for _ in range(14):
    a, b, c = random.randint(12, 99), random.randint(12, 99), random.randint(2, 9)
    Q.append((f"What is {a} * {b} + {c}?", str(a * b + c)))
for _ in range(10):
    a, b = random.randint(100, 999), random.randint(100, 999); Q.append((f"What is {a + b + 1000} - {b}?", str(a + 1000)))
for _ in range(10):
    w = "".join(random.choice("bcdfghklmnprstvz") + random.choice("aeiou") for _ in range(4)); Q.append((f"Reverse the string '{w}'.", w[::-1]))
for _ in range(10):
    xs = random.sample(range(10, 99), 6); Q.append((f"Sort ascending and give the third number: {xs}", str(sorted(xs)[2])))
for _ in range(10):
    w = random.choice(["banana", "mississippi", "committee", "bookkeeper", "assessment", "parallel", "successes", "accommodate", "tomorrow", "possession"]); ch = random.choice(sorted(set(w)))
    Q.append((f"How many times does the letter '{ch}' appear in '{w}'?", str(w.count(ch))))
for _ in range(10):
    n, k = random.randint(20, 60), random.randint(3, 7)
    Q.append((f"Python: what does this print?  x = {n}\nfor i in range({k}):\n    x = x - i if x % 2 else x + i * 2\nprint(x)", str((lambda x: [x := (x - i if x % 2 else x + i * 2) for i in range(k)][-1])(n))))
for _ in range(10):
    s, t1, t2 = random.randint(40, 90), random.randint(2, 6), random.randint(2, 6); Q.append((f"A car drives {t1} hours at {s} km/h, then {t2} hours at {s + 20} km/h. Total distance in km?", str(s * t1 + (s + 20) * t2)))
for _ in range(8):
    n = random.randint(20, 200); Q.append((f"What is {n} in binary?", bin(n)[2:]))
for _ in range(8):
    names = random.sample(["Ava", "Ben", "Cal", "Dee", "Eli"], 4); Q.append((f"{names[0]} is taller than {names[1]}. {names[1]} is taller than {names[2]}. {names[3]} is taller than {names[0]}. Who is shortest?", names[2]))
for _ in range(10):
    a, b = random.randint(2, 12), random.randint(13, 40); x = random.randint(2, 15); Q.append((f"Solve for x: {a}x + {b} = {a * x + b}", str(x)))
model, tok = load(sys.argv[1]); ok = 0; outs = []
for q, a in Q:
    p = tok.apply_chat_template([{"role": "user", "content": q + "\nReply with only the final answer, nothing else."}], add_generation_prompt=True, tokenize=False)
    r = generate(model, tok, p, max_tokens=360).strip(); outs.append(r)
    ans = next((r.split(m)[-1] for m in ("<channel|>", "final<|message|>", "assistantfinal") if m in r), "")          # the answer is what follows the thinking; unfinished thinking = wrong
    got = re.sub(r"[^a-z0-9]", " ", ans.lower().replace(",", "")).split(); ok += a.lower() in got
log(f"QUIZ {sys.argv[2]}: {ok}/{len(Q)} correct"); json.dump(outs, open(os.path.join(HERE, f"quiz_{sys.argv[2]}.json"), "w"))
