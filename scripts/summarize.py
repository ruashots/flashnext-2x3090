#!/usr/bin/env python3
"""Turn the raw jsonl results into medians per (label, context, task)."""
import json, statistics, sys, collections

def load(paths):
    rows = []
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        except FileNotFoundError:
            print(f"missing: {p}", file=sys.stderr)
    return rows

def med(vals):
    vals = [v for v in vals if v]
    return round(statistics.median(vals), 1) if vals else None

def main(paths):
    rows = load(paths)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["label"], r["ctx_target"], r["task"])].append(r)
    print(f"{'configuration':46s} {'ctx':>7s} {'task':6s} {'decode':>8s} {'prefill':>9s} {'wall':>7s} {'n':>3s}")
    print("-" * 92)
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2])):
        rs = g[k]
        print(f"{k[0][:46]:46s} {k[1]:7d} {k[2]:6s} "
              f"{str(med([r['decode_tps'] for r in rs])):>8s} "
              f"{str(med([r['prefill_tps'] for r in rs])):>9s} "
              f"{str(med([r['wall_tps'] for r in rs])):>7s} {len(rs):3d}")

if __name__ == "__main__":
    main(sys.argv[1:])
