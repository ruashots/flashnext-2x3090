#!/bin/bash
# knob.sh <label> [steps...]: prewarm, then bench.py (0/4K/60K, code/reason, 3 runs) and optional steps
L=${1:?usage: knob.sh <label> [quality] [image] [conc]}; shift; S=$(cd "$(dirname "$0")" && pwd); R=${OUT:-$S/../raw}; T=$(mktemp -d); M=${MODEL:-Qwen3.8-Flash-Next-OrcaRouter}; U=${URL:-http://127.0.0.1:8080/v1}
P=$R/strata-0402-probes-2026-10-07.jsonl
python3 $S/bench.py --url $U --model $M --label prewarm-$L --out $T/prewarm-0402.jsonl --contexts 60000 --tasks reason --runs 1 --warmup 0 >/dev/null 2>&1
python3 $S/bench.py --url $U --model $M --label $L --out $R/strata-0402-bench-2026-10-07.jsonl --contexts 0,4000,60000 --tasks code,reason --runs 3 2>&1 | grep ">>"
for s in "$@"; do case $s in
  quality) python3 $S/quality.py --url $U --model $M --label $L --out $R/strata-0402-quality-2026-10-07.jsonl | tail -1 ;;
  image) python3 $S/probe.py image --url $U --label $L --out $P | head -1 ;;
  conc) python3 $S/probe.py concurrent --url $U --label $L --max-tokens 600 --out $P | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('conc', d['r1']['ttft_s'], d['r2']['ttft_s'], d['r1']['decode_tps'], d['r2']['decode_tps'], 'agg', d['aggregate_tps'])" ;;
esac; done
echo KNOB-DONE
