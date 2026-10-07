#!/usr/bin/env python3
"""Export the engine's own per-request timing lines for one server run (one engine start) as JSON rows, and print
the medians used in the 0.1.40.1 -> 0.1.40.2 upgrade report.

    python3 export_engine_timing.py <engine log> "<YYYY-MM-DD HH:MM:SS of the engine start>" <label> >> out.jsonl

<engine log> is Strata's engine log (here /opt/strata/strata-orca-iq4xs.log). A run is every line from its
"[strata] <time> engine started:" line up to the next such line. Per request the engine prints
"strata serve: prompt P tokens = R reused + N read in X ms (...), G generated in Y ms (...), drafts accepted A of D".
Derived: prompt read tok/s = N / X; decode ms per verify step = Y / (G - A) (each step yields its accepted drafts
plus one token). Buckets by prompt size: short < 1,000 tokens, 4k < 10,000, 60k < 100,000. Medians: step time over
requests with G >= 100, prompt read over requests that read more than 1,000 tokens (the same rules as rounds.py)."""
import json, re, statistics, sys
log, start, label = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(log, errors="replace").read().splitlines()
starts = [i for i, l in enumerate(lines) if re.match(r"\[strata\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d engine started:", l)]
i0 = next(i for i in starts if lines[i].startswith(f"[strata] {start} engine started:"))
i1 = next((i for i in starts if i > i0), len(lines))
end = lines[i1][9:28] if i1 < len(lines) else None
ver = next((re.search(r"engine ([0-9.]+)\)", l).group(1) for l in lines[i0:i1] if "session is up (engine" in l), None)
pat = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms.*?, (\d+) generated in (\d+) ms .*?drafts accepted (\d+) of (\d+)")
rows = []
for n, l in enumerate(lines[i0:i1], start=i0 + 1):
    m = pat.search(l)
    if not m:
        continue
    p, reused, read, pms, gen, gms, acc, dr = map(int, m.groups())
    bucket = "short" if p < 1000 else "4k" if p < 10000 else "60k" if p < 100000 else "long"
    row = {"label": label, "engine": ver, "run_started": start, "run_ended": end, "log_line": n, "prompt_tokens": p,
           "reused": reused, "read": read, "read_ms": pms, "read_tok_s": round(read / pms * 1000, 1) if pms else None,
           "generated": gen, "gen_ms": gms, "drafts_accepted": acc, "drafts": dr,
           "ms_per_step": round(gms / (gen - acc), 3) if gen > acc else None, "bucket": bucket}
    rows.append(row); print(json.dumps(row))
for b in ("4k", "60k"):
    st = [r["ms_per_step"] for r in rows if r["bucket"] == b and r["generated"] >= 100 and r["ms_per_step"]]
    pf = [r["read_tok_s"] for r in rows if r["bucket"] == b and r["read"] > 1000]
    print(f"{label} {b}: ms/step median {statistics.median(st):.2f} (n={len(st)}), prompt read median "
          f"{statistics.median(pf):.0f} tok/s (n={len(pf)})", file=sys.stderr)
