#!/usr/bin/env bash
# Unattended final pass.
#   A: fastest configuration that fits at ctx 32768. Prompts capped at 16.5k,
#      which is the largest this configuration is proven to survive.
#   B: one more expert layer on the host and ctx 73728, for every long prompt:
#      the 64k rows, the prefill ladder and the 32k needle cells. A 64k prompt
#      cannot be measured in a 32768-token context, and configuration A OOMs on
#      card 1 at about 18k tokens because the compute buffer grows at run time.
#   C: the smaller quant on A's bands, for the speed and quality curve.
set -uo pipefail
B=/mnt/models/flashnext-bench
OUT=$B/final.log; : > "$OUT"
E='\.ffn_(up|down|gate|gate_up)_(ch|)exps'
BAND_A="blk\.([0-8]|2[5-9]|3[01])$E"
BAND_B="blk\.([0-9]|2[5-9]|3[0-2])$E"
say () { echo "" | tee -a "$OUT"; echo "############ $* :: $(date -Is)" | tee -a "$OUT"; }
alive () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health || true)" = "200" ]; }

say "CONFIG A: bands 0-8 + 25-31, ub 2048, ctx 32768"
bash /root/run_ot.sh "$BAND_A" 32768 UD-IQ4_XS "FINAL-A" -ub 2048 >> "$OUT" 2>&1 \
  || { echo "CONFIG A DID NOT LOAD" | tee -a "$OUT"; }
grep -E "HEALTHY|^0,|^1,|RAM used|CPU_Mapped|CUDA. model buffer" "$OUT" | tail -8
setsid nohup bash /root/sysmon.sh "$B/sysmon-FINAL-A.csv" 4000 >/dev/null 2>&1 </dev/null & disown

if alive; then
  say "A: speed, contexts 0/4k/16k, two tasks, 3 runs"
  python3 /root/bench.py --url http://127.0.0.1:8080/v1 --model x \
    --label "FlashNext UD-IQ4_XS | llama.cpp | 2x3090 | ctx32k" --out "$B/results.jsonl" \
    --contexts 0,4000,16000 --tasks code,reason --runs 3 --warmup 1 --max-tokens 512 >> "$OUT" 2>&1
  grep ">>" "$OUT" | tail -6
fi
if alive; then
  say "A: quality set, non-thinking"
  python3 /root/quality.py --url http://127.0.0.1:8080/v1 --model x \
    --label "FlashNext-IQ4_XS" --out "$B/quality.jsonl" --max-tokens 2048 >> "$OUT" 2>&1
  tail -3 "$OUT"
fi
pkill -f 'sysmo[n].sh' 2>/dev/null
awk -F, 'NR>1{s+=$9;n++;u0+=$4;u1+=$5} END{if(n)printf "A: mean nvme %.0f MB/s, gpu util %.0f%%/%.0f%% over %ds\n",s/n,u0/n,u1/n,n}' "$B/sysmon-FINAL-A.csv" | tee -a "$OUT"

say "CONFIG B: bands 0-9 + 25-32, ub 2048, ctx 73728, every long prompt"
if bash /root/run_ot.sh "$BAND_B" 73728 UD-IQ4_XS "FINAL-B" -ub 2048 >> "$OUT" 2>&1; then
  grep -E "HEALTHY|^0,|^1,|RAM used" "$OUT" | tail -4
  if alive; then
    say "B: speed at 64k"
    python3 /root/bench.py --url http://127.0.0.1:8080/v1 --model x \
      --label "FlashNext UD-IQ4_XS | llama.cpp | 2x3090 | ctx72k" --out "$B/results.jsonl" \
      --contexts 64000 --tasks code,reason --runs 3 --warmup 1 --max-tokens 512 >> "$OUT" 2>&1
    grep ">>" "$OUT" | tail -2
  fi
  if alive; then
    say "B: prefill ladder, wall time for a linear fit"
    python3 /root/qsa_probe.py --mode timing >> "$OUT" 2>&1
    grep -A9 "INSTRUMENT A" "$OUT" | tail -10
  fi
  if alive; then
    say "B: needle grid, 1k control + 16k + 32k"
    python3 /root/qsa_probe.py --mode grid --lengths 1024,16384,32768 --trials 6 > "$B/qsa-grid.log" 2>&1
    tail -32 "$B/qsa-grid.log"
  fi
else
  echo "CONFIG B DID NOT LOAD" | tee -a "$OUT"
fi

say "CONFIG C: UD-Q2_K_XL, A's bands"
if bash /root/run_ot.sh "$BAND_A" 32768 UD-Q2_K_XL "FINAL-C" -ub 2048 >> "$OUT" 2>&1; then
  grep -E "HEALTHY|^0,|^1,|RAM used" "$OUT" | tail -4
  if alive; then
    python3 /root/bench.py --url http://127.0.0.1:8080/v1 --model x \
      --label "FlashNext UD-Q2_K_XL | llama.cpp | 2x3090 | ctx32k" --out "$B/results.jsonl" \
      --contexts 0,4000,16000 --tasks code,reason --runs 2 --warmup 1 --max-tokens 512 >> "$OUT" 2>&1
    grep ">>" "$OUT" | tail -6
  fi
  if alive; then
    python3 /root/quality.py --url http://127.0.0.1:8080/v1 --model x \
      --label "FlashNext-Q2_K_XL" --out "$B/quality.jsonl" --max-tokens 2048 >> "$OUT" 2>&1
  fi
else
  echo "CONFIG C DID NOT LOAD" | tee -a "$OUT"
fi

say "INTEGRITY CHECKS: every count below must be 0"
for f in "$B"/runs/server-FINAL-*.log; do
  echo "--- $f"
  for p in "retrying with smaller batch size" "Compute error" "Context size has been exceeded" \
           "failed to allocate graph" "cudaMalloc failed" "out of memory"; do
    printf '  %-42s %s\n' "$p" "$(grep -c "$p" "$f" 2>/dev/null || echo 0)"
  done
done | tee -a "$OUT"
echo "results.jsonl rows: $(wc -l < $B/results.jsonl)" | tee -a "$OUT"
echo "quality.jsonl rows: $(wc -l < $B/quality.jsonl)" | tee -a "$OUT"
say "FINAL RUN COMPLETE"
