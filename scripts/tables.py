#!/usr/bin/env python3
"""Render the raw results as the markdown tables that go into the write-up."""
import json, statistics, sys, collections

rows = []
for p in sys.argv[1:]:
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line); r["label"] = r["label"].replace("|", "·"); rows.append(r)

def med(rs, k):
    v = [r[k] for r in rs if r.get(k)]
    return round(statistics.median(v), 1) if v else None

def spread(rs, k):
    v = [r[k] for r in rs if r.get(k)]
    return (round(min(v), 1), round(max(v), 1)) if v else (None, None)

g = collections.defaultdict(list)
for r in rows:
    g[(r["label"], r["ctx_target"], r["task"])].append(r)

labels = sorted({k[0] for k in g})
ctxs = sorted({k[1] for k in g})

print("## Decode, tokens per second (median of the runs in each cell)\n")
print("| configuration | " + " | ".join(f"{c or 'short'}" for c in ctxs) + " |")
print("|---" * (len(ctxs) + 1) + "|")
for lab in labels:
    cells = []
    for c in ctxs:
        rs = g.get((lab, c, "code"), []) + g.get((lab, c, "reason"), [])
        cells.append(str(med(rs, "decode_tps") or "-"))
    print(f"| {lab} | " + " | ".join(cells) + " |")

print("\n## Prefill, tokens per second\n")
print("| configuration | " + " | ".join(f"{c or 'short'}" for c in ctxs) + " |")
print("|---" * (len(ctxs) + 1) + "|")
for lab in labels:
    cells = []
    for c in ctxs:
        rs = g.get((lab, c, "code"), []) + g.get((lab, c, "reason"), [])
        cells.append(str(med(rs, "prefill_tps") or "-"))
    print(f"| {lab} | " + " | ".join(cells) + " |")

print("\n## Per cell detail: median decode, min to max, sample count\n")
print("| configuration | ctx | task | decode | spread | n |")
print("|---|---|---|---|---|---|")
for k in sorted(g, key=lambda k: (k[0], k[1], k[2])):
    rs = g[k]
    lo, hi = spread(rs, "decode_tps")
    print(f"| {k[0]} | {k[1]} | {k[2]} | {med(rs,'decode_tps')} | {lo} to {hi} | {len(rs)} |")
