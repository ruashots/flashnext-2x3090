import re, sys, statistics, collections
# usage: python3 scripts/rounds.py < strata-<model>.log   (the engine log Strata writes next to its config)
# per verify round cost from the engine's own per-request lines: rounds = generated - accepted drafts
pat = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms.*?, (\d+) generated in (\d+) ms .*?drafts accepted (\d+) of (\d+)")
b = collections.defaultdict(list); pf = collections.defaultdict(list)
for line in sys.stdin:
    m = pat.search(line)
    if not m: continue
    p, reuse, read, pms, gen, gms, acc, dr = map(int, m.groups())
    bucket = "0" if p < 1000 else "4k" if p < 10000 else "60k" if p < 100000 else "big"
    if gen >= 100:
        rounds = gen - acc
        b[bucket].append(gms / rounds)
    if read > 1000:
        pf[bucket].append(read / pms * 1000)
for k in ["0", "4k", "60k"]:
    v = b[k]
    if v:
        print(f"{k:>4}: ms/round median {statistics.median(v):.2f} (n={len(v)}, min {min(v):.2f}, max {max(v):.2f})", end="")
    if pf[k]:
        print(f"  prefill tok/s median {statistics.median(pf[k]):.0f} (n={len(pf[k])})", end="")
    print()
