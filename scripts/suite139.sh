#!/bin/bash
# suite139.sh <label>: the full check set against a running server.
# URL=http://127.0.0.1:8080/v1 ./scripts/suite139.sh my-run   (MODEL and OUT are optional)
L=${1:?usage: suite139.sh <label>}
S=$(cd "$(dirname "$0")" && pwd); R=${OUT:-$S/../raw}
U=${URL:-http://127.0.0.1:8080/v1}; M=${MODEL:-Qwen3.8-Flash-Next-OrcaRouter}
P=$R/strata-139-probes-2026-10-04.jsonl
python3 $S/bench.py --url $U --model $M --label prewarm-$L --out /dev/null --contexts 60000 --tasks reason --runs 1 --warmup 0 >/dev/null 2>&1
python3 $S/bench.py --url $U --model $M --label $L --out $R/strata-139-bench-2026-10-04.jsonl --contexts 0,4000,60000 --tasks code,reason --runs 3 2>&1 | grep ">>"
python3 $S/quality.py --url $U --model $M --label $L --out $R/strata-139-quality-2026-10-04.jsonl | tail -1
python3 $S/probe.py greedy --url $U --model $M --label $L --out $P
python3 $S/probe.py image --url $U --model $M --label $L --out $P | head -1
python3 $S/probe.py concurrent --url $U --model $M --label $L --max-tokens 600 --out $P
python3 $S/probe.py demote --url $U --model $M --label $L --max-tokens 1500 --offset 5 --out $P
python3 $S/probe.py imgconc --url $U --model $M --label $L --offset 5 --out $P | grep -v "^ \|^#\|^-\|^$\|INVOICE"
python3 $S/probe.py aba --url $U --model $M --label $L --max-tokens 300 --out $P | tail -1
python3 $S/probe.py concurrent --url $U --model $M --label $L --max-tokens 400 --ctx 60000 --out $P
echo SUITE-DONE
