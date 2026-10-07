#!/usr/bin/env python3
"""Harder needle test (RULER-style "multi-key NIAH"): four key -> number facts hidden in the same haystack as Strata's
tools/needle_bench.py, the question asks for ONE key's number. The other three are distractors, so a model that only
spots "the odd sentence" fails. Thinking off, greedy, through the normal API. One JSON line per test.

    python3 needle_multikey.py --strata /path/to/Strata --url http://HOST:8080 --lengths 32k,128k --depths 10,50,90 --out rows.jsonl

The haystack is Strata's own (tools/needle_bench.haystack: the repo's docs and sources in a fixed order), so the text
depends on the Strata checkout: record its commit. Seeds are fixed (7), so the keys and numbers are reproducible.
"""
import argparse, json, random, sys, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--strata", required=True, help="a Strata checkout (its tools/needle_bench.py builds the haystack)")
ap.add_argument("--url", required=True); ap.add_argument("--lengths", required=True); ap.add_argument("--depths", default="10,50,90")
ap.add_argument("--label", required=True); ap.add_argument("--out", required=True); ap.add_argument("--timeout", type=float, default=7200)
ap.add_argument("--stream", action="store_true", help="stream the reply (needed for ~1M-token requests here)")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.strata) / "tools"))
import needle_bench as NB  # noqa: E402

WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern", "meadow",
         "cobalt", "juniper", "tundra", "velvet", "ember", "pebble", "thistle", "marble", "cedar"]
rnd = random.Random(7)
with open(a.out, "a") as fh:
    for L in a.lengths.split(","):
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)
        text = NB.haystack(int(tokens * NB.CHARS_PER_TOKEN))
        for d in (int(x) for x in a.depths.split(",")):
            keys = rnd.sample(WORDS, 8)
            keys = [f"{keys[2*i]}-{keys[2*i+1]}" for i in range(4)]
            nums = [rnd.randint(1000000, 9999999) for _ in range(4)]
            # the asked key at the tested depth, the three distractors at fixed other depths
            spots = [d] + [x for x in (5, 35, 65, 95) if abs(x - d) > 8][:3]
            inserts = sorted(zip(spots, keys, nums), key=lambda t: -t[0])
            body = text
            for depth, k, n in inserts:                       # from the end, so earlier cuts stay valid
                cut = int(len(body) * depth / 100)
                cut = body.rfind("\n", 0, cut) + 1 or cut
                body = body[:cut] + f"\nOne of the special magic numbers for {k} is: {n}.\n" + body[cut:]
            prompt = body + f"\n\nWhat is the special magic number for {keys[0]} mentioned in the text above? Reply with the number only."
            req = {"model": "strata", "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": prompt}]}
            t0 = time.time()
            try:
                if a.stream:
                    req.update(stream=True, stream_options={"include_usage": True})
                r = urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                        data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=a.timeout)
                if a.stream:
                    ans, n_tok = "", None
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
                        try: o = json.loads(line[5:])
                        except ValueError: continue
                        if o.get("usage"): n_tok = o["usage"].get("prompt_tokens")
                        for ch in o.get("choices", []): ans += (ch.get("delta") or {}).get("content") or ""
                else:
                    out = json.loads(r.read()); ans = out["choices"][0]["message"].get("content") or ""; n_tok = out["usage"]["prompt_tokens"]
                row = {"label": a.label, "test": "multikey", "length": L, "depth": d, "prompt_tokens": n_tok,
                       "found": str(nums[0]) in ans, "distractor_given": any(str(x) in ans for x in nums[1:]),
                       "key": keys[0], "number": nums[0], "answer": ans.strip()[:120], "seconds": round(time.time() - t0, 1)}
            except Exception as e:
                row = {"label": a.label, "test": "multikey", "length": L, "depth": d, "error": str(e)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(json.dumps(row), flush=True)
