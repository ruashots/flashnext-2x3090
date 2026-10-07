#!/usr/bin/env python3
"""The 20-prompt quality set (quality.py's PROMPTS) with thinking ON, to check that a server-side guard on
long thinking does not fire on normal requests. Same output format as quality.py, so autograde.py reads it."""
import argparse, json, time, urllib.request
from quality import PROMPTS

def ask(url, model, prompt, max_tokens, timeout=3600):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.load(r)
    msg = obj["choices"][0]["message"]
    return {"answer": msg.get("content") or "", "thinking": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "usage": obj.get("usage"), "seconds": round(time.perf_counter() - t0, 2),
            "finish_reason": obj["choices"][0].get("finish_reason")}

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=12000)
a = ap.parse_args()
with open(a.out, "a") as fh:
    for pid, prompt in PROMPTS:
        try:
            res = ask(a.url, a.model, prompt, a.max_tokens)
        except Exception as e:
            res = {"answer": "", "thinking": "", "error": str(e), "seconds": None}
        fh.write(json.dumps({"label": a.label, "model": a.model, "id": pid, "prompt": prompt, **res}) + "\n"); fh.flush()
        print(f"{pid}: {res.get('seconds')}s think={len(res.get('thinking') or '')} ans={len(res.get('answer') or '')} {res.get('finish_reason')}", flush=True)
