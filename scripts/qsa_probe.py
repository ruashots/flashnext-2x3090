#!/usr/bin/env python3
import argparse, json, random, re, sys, time, urllib.request
from collections import defaultdict

BASE = "http://127.0.0.1:8080"
SENSORS = ["ALTAIR","BOREAS","CYGNUS","DORADO","ERIDANI","FORNAX","GEMINI","HYDRUS",
           "INDUS","JANUS","KEPLER","LYNX","MENSA","NORMA","OCTANS","PYXIS"]

def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])

def ask(prompt, max_tokens=192, cache=True):
    t0 = time.time()
    r = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": 1234,
        "cache_prompt": cache,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    ch = r["choices"][0]["message"]
    out = ch.get("content") or ch.get("reasoning") or ""
    us = r.get("usage", {})
    return out, us.get("prompt_tokens", 0), time.time() - t0

WORDS = ("harbor lantern quarry meadow cinder ledger thicket furnace pebble ravine "
         "compass timber marsh gable orchard bracket cavern trellis mortar shale "
         "beacon foundry heather cobble pasture rafter gully spindle wharf brier "
         "kiln loam vestry paddock chisel bramble sluice granary tallow reef").split()

def filler(rng, n_words):
    out, i = [], 0
    while i < n_words:
        k = rng.randint(8, 20)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        out.append(s[0].upper() + s[1:] + ".")
        i += k
    return " ".join(out)

def build(rng, target_tokens, depths, target_idx):
    recs = []
    names = rng.sample(SENSORS, len(depths))
    for d, nm in zip(depths, names):
        val = f"{rng.randint(10**7, 10**8 - 1)}"
        recs.append({"depth": d, "name": nm, "val": val,
                     "text": f" Maintenance log entry {rng.randint(1000,9999)}: "
                             f"the calibration constant for sensor {nm} is {val}. "})
    probe = filler(rng, 2000)
    tpw = ntok(probe) / 2000.0
    body_tokens = target_tokens - sum(ntok(r["text"]) for r in recs) - 120
    total_words = max(200, int(body_tokens / tpw))
    hay = filler(rng, total_words)
    for _ in range(6):
        got = ntok(hay)
        if abs(got - body_tokens) <= max(64, body_tokens * 0.01): break
        total_words = max(200, int(total_words * body_tokens / max(got, 1)))
        hay = filler(rng, total_words)
    pieces, prev = [], 0
    for r in sorted(recs, key=lambda x: x["depth"]):
        cut = int(len(hay) * r["depth"])
        cut = hay.find(" ", cut)
        cut = len(hay) if cut == -1 else cut
        pieces.append(hay[prev:cut]); pieces.append(r["text"]); prev = cut
    pieces.append(hay[prev:])
    tgt = recs[target_idx]
    prompt = ("Read the maintenance report below, then answer the question at the end.\n\n"
              "=== BEGIN REPORT ===\n" + "".join(pieces) + "\n=== END REPORT ===\n\n"
              f"Question: What is the calibration constant for sensor {tgt['name']}?\n"
              "Reply with only the number. If the report does not contain it, reply exactly NOT_PRESENT.")
    return prompt, tgt, recs

def grade(out, tgt, recs):
    digits = re.findall(r"\d{6,9}", out)
    if tgt["val"] in digits: return "EXACT", ""
    for r in recs:
        if r is not tgt and r["val"] in digits:
            return "WRONG_RECORD", f"took depth {r['depth']:.2f}"
    if "NOT_PRESENT" in out.upper() or not digits: return "ABSENT", ""
    for d in digits:
        if len(d) == len(tgt["val"]) and sum(a == b for a, b in zip(d, tgt["val"])) >= 5:
            return "CORRUPT", d
    return "ABSENT", (digits[0] if digits else "")

def timing():
    print("\n=== INSTRUMENT A: prefill scaling (cache OFF) ===")
    print("Flat tok/s past 8k => sparse budget engaged. ~1/N falloff => DENSE fallback.\n")
    rng = random.Random(7)
    print(f"{'ctx':>8} {'prompt_tok':>11} {'prefill tok/s':>14} {'vs 2k':>8}")
    base = None
    for L in (2048, 8192, 16384, 32768, 65536):
        p, _, _ = build(rng, L, [0.5], 0)
        try:
            _, ptok, dt = ask(p, max_tokens=1, cache=False)
        except Exception as e:
            print(f"{L:>8} FAILED: {e}"); continue
        rate = ptok / dt
        base = base or rate
        print(f"{L:>8} {ptok:>11} {rate:>14.1f} {rate/base:>7.2f}x", flush=True)

def grid(lengths, trials, depths):
    print("\n=== INSTRUMENT B: needle grid (5 records, 1 asked, distractors elsewhere) ===")
    res = defaultdict(lambda: defaultdict(int)); notes = defaultdict(list)
    for L in lengths:
        for di, d in enumerate(depths):
            for t in range(trials):
                rng = random.Random(hash((L, di, t)) & 0xffffffff)
                p, tgt, recs = build(rng, L, depths, di)
                try:
                    out, ptok, dt = ask(p)
                except Exception as e:
                    res[(L, d)]["ERROR"] += 1; print(f"  L={L} d={d} t={t} ERROR {e}"); continue
                v, note = grade(out, tgt, recs)
                res[(L, d)][v] += 1
                if note: notes[(L, d)].append(note)
                print(f"  L={L:<6} depth={d:<5} trial={t:<3} {v:<13} ptok={ptok}", flush=True)
    print("\n=== RESULTS: EXACT / total per cell ===")
    print(f"{'ctx':>8} " + " ".join(f"{('d=%.2f' % d):>9}" for d in depths))
    for L in lengths:
        row = f"{L:>8} "
        for d in depths:
            c = res[(L, d)]; n = sum(c.values())
            row += f"{c['EXACT']:>5}/{n:<3}"
        print(row)
    print("\n=== FAILURE BREAKDOWN (non-EXACT only) ===")
    for L in lengths:
        for d in depths:
            c = res[(L, d)]
            bad = {k: v for k, v in c.items() if k != "EXACT" and v}
            if bad:
                extra = ("  " + "; ".join(notes[(L, d)][:3])) if notes[(L, d)] else ""
                print(f"  L={L:<6} depth={d:<5} {bad}{extra}")
    verdict(res, lengths, depths, trials)

def verdict(res, lengths, depths, trials):
    print("\n=== VERDICT ===")
    short = min(lengths)
    sc = sum(res[(short, d)]["EXACT"] for d in depths)
    sn = sum(sum(res[(short, d)].values()) for d in depths)
    if sn and sc / sn < 0.95:
        print(f"ABORT: positive control failed at {short} ({sc}/{sn}). The quant or the")
        print("harness is broken, not long context. Fix that before reading anything else.")
        return
    print(f"Positive control at {short}: {sc}/{sn} PASS.")
    red = False
    for L in lengths:
        if L <= 4096: continue
        deep = res[(L, max(depths))]["EXACT"]
        for d in depths:
            e = res[(L, d)]["EXACT"]
            if deep >= 0.9 * trials and e <= 0.5 * trials:
                print(f"RED: L={L} depth={d} scores {e}/{trials} while depth={max(depths)}"
                      f" scores {deep}/{trials}. Position-dependent. Retrieval is broken.")
                red = True
    if not red:
        worst = min((res[(L, d)]["EXACT"] for L in lengths for d in depths), default=0)
        agg = sum(res[(L, d)]["CORRUPT"] for L in lengths for d in depths)
        if worst >= 0.9 * trials:
            print("GREEN on retrieval: every cell >= 90%. NOTE: only meaningful alongside")
            print("Instrument A. If prefill showed dense fallback, you proved the dense path")
            print("works and learned nothing about QSA. Report both together.")
        elif agg and agg >= 0.3 * trials:
            print("AMBER: failures position-independent, dominated by CORRUPT. That is")
            print("quantization damage, not retrieval. Go up a quant.")
        else:
            print(f"AMBER: worst cell {worst}/{trials}, no clean position signature.")
            print("Raise trials to 30 before publishing anything about this cell.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timing","grid","both"], default="both")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--lengths", default="1024,4096,16384,32768")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--depths", default="0.05,0.25,0.50,0.75,0.95")
    a = ap.parse_args()
    BASE = a.base
    L = [int(x) for x in a.lengths.split(",")]
    D = [float(x) for x in a.depths.split(",")]
    if a.mode in ("timing","both"): timing()
    if a.mode in ("grid","both"): grid(L, a.trials, D)
