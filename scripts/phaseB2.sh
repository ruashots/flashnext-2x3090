#!/usr/bin/env bash
# Phase B retry with a smaller ubatch. On this architecture the QSA compute
# buffer scales with n_kv x ubatch: the bias input is [n_kv, n_tps, n_stream],
# so at ctx 73728 with -ub 2048 that one tensor is 576 MiB against 256 MiB at
# ctx 32768. Halving ubatch halves every n_kv-scaled input, which is the term
# that grows with the long prompts this phase exists to measure.
set -uo pipefail
B=/mnt/models/flashnext-bench
OUT=$B/phaseB2.log; : > "$OUT"
E='\.ffn_(up|down|gate|gate_up)_(ch|)exps'
say () { echo "" | tee -a "$OUT"; echo "############ $* :: $(date -Is)" | tee -a "$OUT"; }
alive () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health || true)" = "200" ]; }

for TRY in "0-9+25-32:blk\.([0-9]|2[5-9]|3[0-2])$E:1024" \
           "0-11+25-34:blk\.(([0-9])|1[01]|2[5-9]|3[0-4])$E:1024"; do
  DESC="${TRY%%:*}"; REST="${TRY#*:}"; BAND="${REST%:*}"; UB="${REST##*:}"
  say "phase B retry, bands $DESC, ub $UB, ctx 73728"
  if bash /root/run_ot.sh "$BAND" 73728 UD-IQ4_XS "FINAL-B2-$DESC" -ub "$UB" >> "$OUT" 2>&1; then
    grep -E "HEALTHY|^0,|^1,|RAM used|compute buffer" "$OUT" | tail -6
    if alive; then
      say "B2: speed at 64k"
      python3 /root/bench.py --url http://127.0.0.1:8080/v1 --model x \
        --label "FlashNext UD-IQ4_XS | 2x3090 | ctx72k ub$UB" --out "$B/results.jsonl" \
        --contexts 64000 --tasks code,reason --runs 3 --warmup 1 --max-tokens 512 >> "$OUT" 2>&1
      grep ">>" "$OUT" | tail -2
    fi
    if alive; then
      say "B2: prefill ladder"
      python3 /root/qsa_probe.py --mode timing >> "$OUT" 2>&1
      grep -A9 "INSTRUMENT A" "$OUT" | tail -10
    fi
    if alive; then
      say "B2: needle grid 1k / 16k / 32k"
      python3 /root/qsa_probe.py --mode grid --lengths 1024,16384,32768 --trials 6 > "$B/qsa-grid.log" 2>&1
      tail -32 "$B/qsa-grid.log"
    fi
    break
  else
    echo "bands $DESC ub $UB DID NOT LOAD" | tee -a "$OUT"
    grep -iE "out of memory|allocating" "$B/runs/server-FINAL-B2-$DESC.log" 2>/dev/null | tail -2 | tee -a "$OUT"
  fi
done
say "PHASE B2 COMPLETE"
