#!/usr/bin/env python3
"""Strata's tools/needle_bench.py, the same haystack, needle sentence, question and seeded code words, but the request
is STREAMED. Used for the 1M setup: there a non-streamed reply to a ~1M-token request was never delivered (the engine
finished in 544 s; the client's connection stayed open with no reply for 30 more minutes). JSON lines with --out."""
import argparse, json, random, sys, time, urllib.request
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--strata", required=True); ap.add_argument("--url", required=True); ap.add_argument("--lengths", required=True)
ap.add_argument("--depths", default="10,50,90"); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--skip", type=int, default=0, help="draw (and skip) this many code words first, to continue a run")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.strata) / "tools")); import needle_bench as NB  # noqa: E402
rnd = random.Random(7)
for _ in range(a.skip): rnd.choice(NB.WORDS); rnd.choice(NB.WORDS); rnd.randint(100, 999)
with open(a.out, "a") as fh:
    for L in a.lengths.split(","):
        tokens = int(int(float(L.lower().rstrip("k")) * 1024) * 0.98)
        text = NB.haystack(int(tokens * NB.CHARS_PER_TOKEN))
        for d in (int(x) for x in a.depths.split(",")):
            word = f"{rnd.choice(NB.WORDS)}-{rnd.choice(NB.WORDS)}-{rnd.randint(100, 999)}"
            cut = int(len(text) * d / 100); cut = text.rfind("\n", 0, cut) + 1 or cut
            prompt = (text[:cut] + f"\nThe secret code word for this text is: {word}. Remember it.\n" + text[cut:] +
                      "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
            body = {"model": "strata", "max_tokens": 40, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
            t0 = time.time(); ans, n = "", None
            with urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
                    try: o = json.loads(line[5:])
                    except ValueError: continue
                    if o.get("usage"): n = o["usage"].get("prompt_tokens")
                    for ch in o.get("choices", []): ans += (ch.get("delta") or {}).get("content") or ""
            row = {"label": a.label, "test": "needle", "length": L, "depth": d, "prompt_tokens": n, "seconds": round(time.time() - t0, 1),
                   "found": word in ans, "word": word, "answer": ans.strip()[:200], "streamed": True}
            fh.write(json.dumps(row) + "\n"); fh.flush(); print(json.dumps(row), flush=True)
