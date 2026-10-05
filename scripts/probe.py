#!/usr/bin/env python3
"""Extra probes beside bench.py and quality.py: greedy identity, image, two at once, demotion.
Every mode appends JSON lines to --out with --label."""
import argparse, base64, json, sys, threading, time, uuid, urllib.request
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from bench import build_filler, TASKS

IMG = __import__("os").path.join(__import__("os").path.dirname(__file__), "probe-test.png")


def post(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(url, body, stamps, timeout=1800):
    """Streams; appends (t, n_chars) per content piece to stamps; returns (text, usage)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    text, usage, buf = [], None, b""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read1(2048)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    continue
                try:
                    o = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                for ch in o.get("choices", []):
                    d = ch.get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                    if piece:
                        stamps.append(time.perf_counter())
                    if d.get("content"):
                        text.append(d["content"])   # the answer only, as a client keeps it in the history
    return "".join(text), usage


GREEDY = [
    ("g-code", TASKS["code"]),
    ("g-reason", TASKS["reason"]),
    ("g-4k", build_filler(4000) + "\n\n" + TASKS["recall"]),
    ("g-es", "Explica en tres frases por que el cielo es azul."),
]


def greedy(a, fh):
    for pid, prompt in GREEDY:
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        o = post(a.url, body)
        rec = {"label": a.label, "mode": "greedy", "id": pid, "answer": o["choices"][0]["message"].get("content") or "",
               "usage": o.get("usage"), "seconds": round(time.perf_counter() - t0, 2)}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(pid, rec["seconds"], len(rec["answer"]), flush=True)


def image(a, fh):
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    q = ("Read the text in this image exactly, then list every shape with its colour and whether it is on the "
         "left or the right.")
    body = {"model": a.model, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": q}]}]}
    t0 = time.perf_counter()
    o = post(a.url, body)
    ans = o["choices"][0]["message"].get("content") or ""
    low = ans.lower()
    checks = {"4729": "4729" in ans, "invoice": "invoice" in low, "red": "red" in low, "blue": "blue" in low,
              "circle": "circle" in low, "square": ("square" in low or "rectangle" in low)}
    rec = {"label": a.label, "mode": "image", "answer": ans, "checks": checks, "pass": all(checks.values()),
           "usage": o.get("usage"), "seconds": round(time.perf_counter() - t0, 2)}
    fh.write(json.dumps(rec) + "\n"); fh.flush()
    print(json.dumps({k: rec[k] for k in ("checks", "pass", "seconds")}), "\n", ans[:400], flush=True)


def rates(stamps, t_start, usage, chars_total=None):
    if not stamps:
        return {}
    ttft = stamps[0] - t_start
    n = (usage or {}).get("completion_tokens") or len(stamps)
    dec = (n / (stamps[-1] - stamps[0])) if stamps[-1] > stamps[0] else None
    return {"ttft_s": round(ttft, 3), "tokens": n, "decode_tps": round(dec, 2) if dec else None,
            "wall_s": round(stamps[-1] - t_start, 2), "prompt_tokens": (usage or {}).get("prompt_tokens")}


def concurrent(a, fh):
    """Two requests started together (offset --offset s), both with a thinking-on task, max_tokens each."""
    res = {}

    def one(name, delay, ctx, task):
        time.sleep(delay)
        prompt = "Session " + uuid.uuid4().hex + ".\n" + ((build_filler(ctx) + "\n\n") if ctx else "") + TASKS[task]
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
                "temperature": 0.7, "top_p": 0.95}
        st = []
        t0 = time.perf_counter()
        _, usage = stream(a.url, body, st)
        res[name] = (t0, st, usage)

    th = [threading.Thread(target=one, args=("r1", 0, a.ctx, "code")),
          threading.Thread(target=one, args=("r2", a.offset, a.ctx, "reason"))]
    T0 = time.perf_counter()
    for t in th: t.start()
    for t in th: t.join()
    out = {"label": a.label, "mode": "concurrent", "ctx": a.ctx, "offset_s": a.offset, "max_tokens": a.max_tokens}
    total_tok = 0
    for k, (t0, st, usage) in sorted(res.items()):
        out[k] = rates(st, t0, usage)
        total_tok += out[k].get("tokens") or 0
    end = max(st[-1] for (_, st, _) in res.values() if st)
    out["total_wall_s"] = round(end - T0, 2)
    out["aggregate_tps"] = round(total_tok / (end - T0), 2)
    # aggregate decode while both were generating
    s1, s2 = res["r1"][1], res["r2"][1]
    lo, hi = max(s1[0], s2[0]), min(s1[-1], s2[-1])
    if hi > lo:
        both = sum(1 for x in s1 if lo <= x <= hi) + sum(1 for x in s2 if lo <= x <= hi)
        out["both_streaming_s"] = round(hi - lo, 2)
        out["both_streaming_chunks_per_s"] = round(both / (hi - lo), 2)
    fh.write(json.dumps(out) + "\n"); fh.flush()
    print(json.dumps(out), flush=True)


def demote(a, fh):
    """A long request alone; a short one arrives after --offset s; A's chunk rate before, during, after."""
    res = {}

    def one(name, delay, task, mt):
        time.sleep(delay)
        prompt = "Session " + uuid.uuid4().hex + ".\n" + TASKS[task]
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
                "temperature": 0.7, "top_p": 0.95}
        st = []
        t0 = time.perf_counter()
        _, usage = stream(a.url, body, st)
        res[name] = (t0, st, usage)

    th = [threading.Thread(target=one, args=("long", 0, "code", a.max_tokens)),
          threading.Thread(target=one, args=("short", a.offset, "reason", 200))]
    for t in th: t.start()
    for t in th: t.join()
    la, sa = res["long"][1], res["short"][1]
    s_start, s_end = res["short"][0], sa[-1]

    def rate(lo, hi):
        n = sum(1 for x in la if lo <= x < hi)
        return round(n / (hi - lo), 2) if hi > lo else None
    out = {"label": a.label, "mode": "demote", "offset_s": a.offset,
           "long": rates(la, res["long"][0], res["long"][2]), "short": rates(sa, s_start, res["short"][2]),
           "long_chunks_per_s_before": rate(la[0], s_start), "long_chunks_per_s_during": rate(s_start, s_end),
           "long_chunks_per_s_after": rate(s_end, la[-1]) if la[-1] > s_end else None}
    fh.write(json.dumps(out) + "\n"); fh.flush()
    print(json.dumps(out), flush=True)


def imgconc(a, fh):
    """An image request sent while a long text request is decoding: the image answer must still be right."""
    res = {}

    def longreq():
        prompt = "Session " + uuid.uuid4().hex + ".\n" + TASKS["code"]
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200,
                "temperature": 0.7, "top_p": 0.95}
        st = []
        t0 = time.perf_counter()
        _, usage = stream(a.url, body, st)
        res["long"] = rates(st, t0, usage)
    th = threading.Thread(target=longreq); th.start()
    time.sleep(a.offset or 5)
    a2 = argparse.Namespace(**vars(a)); a2.label = a.label + "-during-decode"
    image(a2, fh)
    th.join()
    out = {"label": a.label, "mode": "imgconc", "long": res.get("long")}
    fh.write(json.dumps(out) + "\n"); fh.flush()
    print(json.dumps(out), flush=True)


def aba(a, fh):
    """Two long conversations, A then B then A's next turn: the TTFT of each (does A come back without a re-read?)."""
    def doc(tag):
        return (f"Document {tag} ({uuid.uuid4().hex}). " + build_filler(a.ctx or 60000))
    convs = {"A": [{"role": "user", "content": doc("A") + "\n\n" + TASKS["recall"]}],
             "B": [{"role": "user", "content": doc("B") + "\n\n" + TASKS["recall"]}]}
    out = {"label": a.label, "mode": "aba", "ctx": a.ctx or 60000}
    for step, name in [("A1", "A"), ("B1", "B"), ("A2", "A")]:
        if step == "A2":
            convs["A"] += [{"role": "user", "content": "Now say in one sentence what happens to a record that fails validation."}]
        body = {"model": a.model, "messages": convs[name], "max_tokens": a.max_tokens, "temperature": 0.7, "top_p": 0.95}
        st = []
        t0 = time.perf_counter()
        text, usage = stream(a.url, body, st)
        out[step] = rates(st, t0, usage)
        if step != "A2":
            convs[name] = convs[name] + [{"role": "assistant", "content": text}]
        print(step, json.dumps(out[step]), flush=True)
    fh.write(json.dumps(out) + "\n"); fh.flush()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["greedy", "image", "concurrent", "demote", "imgconc", "aba"])
    ap.add_argument("--url", required=True); ap.add_argument("--model", default="Qwen3.8-Flash-Next-OrcaRouter")
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=0); ap.add_argument("--offset", type=float, default=0.0)
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        {"greedy": greedy, "image": image, "concurrent": concurrent, "demote": demote, "imgconc": imgconc, "aba": aba}[a.mode](a, fh)
