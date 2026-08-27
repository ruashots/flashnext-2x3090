#!/usr/bin/env bash
# Phase D: UD-Q2_K_XL on bands sized for its own expert pool.
# Its routed experts are 42.92 GiB against IQ4_XS's 55.43, so it needs far fewer
# layers on the host. Running it on IQ4_XS's bands leaves VRAM idle and adds CPU
# matmuls, which would understate the quant rather than measure it.
set -uo pipefail
B=/mnt/models/flashnext-bench
OUT=$B/phaseD.log; : > "$OUT"
E='\.ffn_(up|down|gate|gate_up)_(ch|)exps'
say () { echo "" | tee -a "$OUT"; echo "############ $* :: $(date -Is)" | tee -a "$OUT"; }
alive () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health || true)" = "200" ]; }

for BAND_DESC in "0-3+25-28:blk\.([0-3]|2[5-8])$E" "0-5+25-30:blk\.([0-5]|2[5-9]|30)$E"; do
  DESC="${BAND_DESC%%:*}"; BAND="${BAND_DESC#*:}"
  say "Q2_K_XL bands $DESC, ub 2048, ctx 32768"
  if bash /root/run_ot.sh "$BAND" 32768 UD-Q2_K_XL "FINAL-D-$DESC" -ub 2048 >> "$OUT" 2>&1; then
    grep -E "HEALTHY|^0,|^1,|RAM used|CUDA. model buffer|CPU_Mapped" "$OUT" | tail -6
    if alive; then
      python3 /root/bench.py --url http://127.0.0.1:8080/v1 --model x \
        --label "FlashNext UD-Q2_K_XL own bands $DESC | 2x3090 | ctx32k" --out "$B/results.jsonl" \
        --contexts 0,4000,16000 --tasks code,reason --runs 2 --warmup 1 --max-tokens 512 >> "$OUT" 2>&1
      grep ">>" "$OUT" | tail -6
      break   # first band that loads and serves is the one we report
    fi
  else
    echo "bands $DESC DID NOT LOAD, trying a wider band" | tee -a "$OUT"
    grep -iE "out of memory|allocating" "$B/runs/server-FINAL-D-$DESC.log" 2>/dev/null | tail -2 | tee -a "$OUT"
  fi
done

if alive; then
  say "D: quality set on the winning Q2_K_XL bands"
  python3 /root/quality.py --url http://127.0.0.1:8080/v1 --model x \
    --label "FlashNext-Q2_K_XL-own-bands" --out "$B/quality.jsonl" --max-tokens 2048 >> "$OUT" 2>&1
fi
say "PHASE D COMPLETE"
