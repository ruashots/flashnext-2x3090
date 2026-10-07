#!/usr/bin/env python3
"""Markdown tables from the canonical benchmark rows (raw/strata-canonical-*-2026-10-07.jsonl)."""
import collections, json, statistics, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "raw"
def rows(name):
    p = R / f"strata-canonical-{name}-2026-10-07.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
med = lambda v: statistics.median(v) if v else None
f0 = lambda x: "-" if x is None else f"{x:,.0f}"
f1 = lambda x: "-" if x is None else f"{x:,.1f}"

print("## Speed ladder (one request; prompt read and answer written, tokens/s, median of runs)\n")
print("| setup | context target | prompt tokens | prompt read tok/s | answer tok/s | runs |\n|---|---:|---:|---:|---:|---:|")
g = collections.defaultdict(list)
for r in rows("bench"): g[(r["label"], r["ctx_target"])].append(r)
for (lab, ctx), v in sorted(g.items(), key=lambda kv: (kv[0][0] != "C262", kv[0][0] != "C512", kv[0][1])):
    print(f"| {lab} | {ctx:,} | {f0(med([x['prompt_tokens'] for x in v]))} | {f0(med([x['prefill_tps'] for x in v if ctx]))} | "
          f"{f1(med([x['decode_tps'] for x in v]))} | {len(v)} |")

print("\n## One vs two requests at once (600 tokens each)\n")
print("| context | 1st request first token s | 2nd request first token s | each tok/s | both together tok/s |\n|---:|---:|---:|---:|---:|")
g = collections.defaultdict(list)
for r in rows("probes"):
    if r.get("mode") == "concurrent": g[r["ctx"]].append(r)
for ctx, v in sorted(g.items()):
    print(f"| {ctx:,} | {f1(med([x['r1']['ttft_s'] for x in v]))} | {f1(med([x['r2']['ttft_s'] for x in v]))} | "
          f"{f1(med([(x['r1']['decode_tps']+x['r2']['decode_tps'])/2 for x in v]))} | {f1(med([x['aggregate_tps'] for x in v]))} ({len(v)} runs) |")

for test, title in (("needle", "Needle (one hidden code word, Strata's tools/needle_bench.py)"),
                    ("multikey", "Multi-key needle (four key/number facts, asked for one)")):
    print(f"\n## {title}\n")
    g = collections.defaultdict(dict)
    for r in rows("needle"):
        if r.get("test") == test and "found" in r: g[(r["label"], r["length"])][r["depth"]] = r
    depths = sorted({d for v in g.values() for d in v})
    print("| setup | length | prompt tokens | " + " | ".join(f"depth {d}%" for d in depths) + " |")
    print("|---|---|---:|" + "---|" * len(depths))
    for (lab, L), v in g.items():
        toks = med([x["prompt_tokens"] for x in v.values()])
        print(f"| {lab} | {L} | {f0(toks)} | " + " | ".join(("found" if v[d]["found"] else "MISSED") if d in v else "" for d in depths) + " |")

q = rows("gsm8k")
if q:
    print("\n## GSM8K (0-shot, greedy, thinking off)\n")
    for lab in sorted({r["label"] for r in q}):
        v = [r for r in q if r["label"] == lab]
        print(f"- {lab}: {sum(r['ok'] for r in v)} of {len(v)} correct = {100*sum(r['ok'] for r in v)/len(v):.1f}%")
