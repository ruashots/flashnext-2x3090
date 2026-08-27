#!/usr/bin/env python3
"""
Same-harness benchmark for OpenAI-compatible servers (vLLM and llama-server).
Measures TTFT, decode tok/s and end-to-end wall tok/s on identical prompts.
Writes one JSON line per run to the output file.
"""
import argparse, json, os, statistics, sys, time, uuid
import urllib.request

def build_filler(approx_tokens):
    """Deterministic filler text of roughly the requested token count."""
    unit = ("The service reads a batch of records from the queue, validates each field "
            "against the schema, writes the accepted rows to the primary table and pushes "
            "the rejected rows to a dead letter topic with the reason attached. ")
    # ~40 tokens per unit, measured empirically for this tokenizer family
    n = max(1, approx_tokens // 40)
    return unit * n

TASKS = {
    "code": "Write a Python function `merge_intervals(intervals)` that merges overlapping "
            "closed intervals and returns them sorted. Handle empty input, single interval, "
            "full containment and touching endpoints. Include the docstring and three asserts.",
    "reason": "A train leaves station A at 09:00 travelling at 80 km/h. A second train leaves "
              "station B, 400 km away, at 09:30 travelling toward A at 120 km/h. At what clock "
              "time do they meet, and how far from A? Show each step.",
    "recall": "Read the text above. Name the two destinations a record can reach, and state the "
              "single condition that decides which one it goes to. Answer in two sentences.",
}

def run_one(base_url, model, prompt, max_tokens, temperature, timeout):
    # leading nonce so no server-side prefix cache can serve this prompt
    prompt = "Session " + uuid.uuid4().hex + ".\n" + prompt
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    text_len = 0
    buf = b""
    last_tok_t = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read1(2048)          # read1: returns as soon as bytes are available
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    buf = b""
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        last_tok_t = now
                        n_chunks += 1
                        text_len += len(piece)
    total = time.perf_counter() - t0
    out_tok = (usage or {}).get("completion_tokens") or n_chunks
    in_tok = (usage or {}).get("prompt_tokens")
    decode_s = (last_tok_t - t0 - ttft) if (last_tok_t and ttft) else (total - (ttft or 0))
    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "total_s": round(total, 4),
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "decode_tps": round(out_tok / decode_s, 2) if decode_s > 0 and out_tok else None,
        "wall_tps": round(out_tok / total, 2) if out_tok else None,
        "prefill_tps": round(in_tok / ttft, 1) if in_tok and ttft else None,
        "chars": text_len,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--contexts", default="0,4000,16000,64000")
    ap.add_argument("--tasks", default="code,reason")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # One long generation before anything is measured. With experts held in host
    # page cache, a cold run measures cache misses rather than the model, and a
    # short prompt never warms the cache the way a long prefill does. Every number
    # below is therefore a warm-cache number, stated as such.
    for _ in range(2):
        try:
            run_one(args.url, args.model, build_filler(3000) + "\n\n" + TASKS["code"],
                    512, args.temperature, args.timeout)
        except Exception as e:
            print(f"[prewarm fail] {e}", flush=True)

    with open(args.out, "a") as fh:
        for ctx in contexts:
            filler = build_filler(ctx) if ctx else ""
            for task in tasks:
                if ctx and task == "code":
                    prompt = filler + "\n\n" + TASKS[task]
                elif ctx:
                    prompt = filler + "\n\n" + TASKS["recall"]
                else:
                    prompt = TASKS[task]
                for w in range(args.warmup):
                    try:
                        run_one(args.url, args.model, prompt, 64, args.temperature, args.timeout)
                    except Exception as e:
                        print(f"[warmup fail] ctx={ctx} task={task}: {e}", flush=True)
                samples = []
                for r in range(args.runs):
                    try:
                        res = run_one(args.url, args.model, prompt, args.max_tokens,
                                      args.temperature, args.timeout)
                    except Exception as e:
                        print(f"[fail] ctx={ctx} task={task} run={r}: {e}", flush=True)
                        continue
                    res.update({"label": args.label, "model": args.model,
                                "ctx_target": ctx, "task": task, "run": r,
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    samples.append(res)
                    print(json.dumps(res), flush=True)
                if samples:
                    dec = [s["decode_tps"] for s in samples if s["decode_tps"]]
                    pre = [s["prefill_tps"] for s in samples if s["prefill_tps"]]
                    print(f"  >> {args.label} ctx~{ctx} {task}: "
                          f"decode median {statistics.median(dec) if dec else 'na'} tok/s, "
                          f"prefill median {statistics.median(pre) if pre else 'na'} tok/s",
                          flush=True)

if __name__ == "__main__":
    main()
