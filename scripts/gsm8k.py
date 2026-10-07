#!/usr/bin/env python3
"""GSM8K test set (all 1,319 problems), 0-shot, greedy, thinking off, through an OpenAI-compatible server.
Data: openai/grade-school-math test.jsonl (sha256 3730d312...b3c39d14), downloaded on first use.
Scored by exact match of the final number (the model is asked to end with '#### <number>'; without it, the last
number in the reply is taken). --workers 2 sends two requests at once (the answers are the same as one at a time on
a server whose batch slots are exact; see docs/BATCHING.md)."""
import argparse, hashlib, json, re, threading, time, urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
PROMPT = ("Solve the problem. Think step by step, then give the final answer on the last line as '#### <number>'.\n\n"
          "Problem: {q}")

def number(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", s) or re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
data = Path(__file__).with_name("gsm8k_test.jsonl")
if not data.exists():
    data.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
assert hashlib.sha256(data.read_bytes()).hexdigest() == SHA, "gsm8k test.jsonl does not match the pinned file"
items = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
if a.limit: items = items[:a.limit]
lock, rows, it = threading.Lock(), [], iter(enumerate(items))

def worker():
    while True:
        with lock:
            try: i, ex = next(it)
            except StopIteration: return
        body = {"model": a.model, "messages": [{"role": "user", "content": PROMPT.format(q=ex["question"])}],
                "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        try:
            o = json.load(urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
            ans = o["choices"][0]["message"].get("content") or ""
        except Exception as e:
            ans = f"ERROR {e}"
        gold = float(ex["answer"].split("####")[-1].replace(",", "").strip())
        got = number(ans)
        row = {"label": a.label, "i": i, "gold": gold, "got": got, "ok": got is not None and abs(got - gold) < 1e-6,
               "seconds": round(time.time() - t0, 2), "answer": ans[-300:]}
        with lock:
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"{len(rows)}/{len(items)}  accuracy so far {sum(r['ok'] for r in rows)/len(rows):.3f}", flush=True)

th = [threading.Thread(target=worker) for _ in range(a.workers)]
t0 = time.time()
for t in th: t.start()
for t in th: t.join()
rows.sort(key=lambda r: r["i"])
with open(a.out, "a") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
acc = sum(r["ok"] for r in rows) / len(rows)
print(json.dumps({"label": a.label, "n": len(rows), "accuracy": round(acc, 4), "errors": sum(r["answer"].startswith("ERROR") for r in rows),
                  "wall_min": round((time.time() - t0) / 60, 1)}))
