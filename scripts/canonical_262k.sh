#!/bin/bash
# URL=http://127.0.0.1:8080 STRATA=/path/to/Strata-v0.1.40.2 scripts/canonical_262k.sh
# The canonical benchmark set on the tuned 262K two-at-once config (Strata 0.1.40.2). Rows go to raw/ with label prefix C262.
S=$(cd "$(dirname "$0")" && pwd); R=${OUT:-$S/../raw}; T=$(mktemp -d); M=${MODEL:-Qwen3.8-Flash-Next-OrcaRouter}; U=${URL:-http://127.0.0.1:8080}
STRATA=${STRATA:?set STRATA to a plain clone of Strata at v0.1.40.2}   # its tools/needle_bench.py builds the haystack
B=$R/strata-canonical-bench-2026-10-07.jsonl; P=$R/strata-canonical-probes-2026-10-07.jsonl; N=$R/strata-canonical-needle-2026-10-07.jsonl
python3 $S/bench.py --url $U/v1 --model $M --label prewarm --out $T/prewarm-c.jsonl --contexts 60000 --tasks reason --runs 1 --warmup 0 >/dev/null 2>&1
echo "== A1 speed ladder"; python3 $S/bench.py --url $U/v1 --model $M --label C262 --out $B --contexts 0,4000,16000,32000,64000,128000 --tasks code,reason --runs 3 2>&1 | grep ">>"
python3 $S/bench.py --url $U/v1 --model $M --label C262 --out $B --contexts 200000,245000 --tasks reason --runs 1 --warmup 0 2>&1 | grep ">>"
echo "== A2 one vs two at once"; for c in 0 0 0 4000 4000 4000 32000; do python3 $S/probe.py concurrent --url $U/v1 --label C262 --ctx $c --max-tokens 600 --out $P | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('ctx',d['ctx'],'ttft',d['r1']['ttft_s'],d['r2']['ttft_s'],'tps',d['r1']['decode_tps'],d['r2']['decode_tps'],'agg',d['aggregate_tps'])"; done
echo "== A3 needle"; python3 $STRATA/tools/needle_bench.py --url $U --lengths 4k,16k,32k,64k,128k,200k,256k --depths 10,25,50,75,90 --out $T/needle262.json
python3 -c "import json; [print(json.dumps({'label':'C262','test':'needle',**r})) for r in json.load(open('$T/needle262.json'))]" >> $N
echo "== A4 multikey"; python3 $S/needle_multikey.py --strata $STRATA --url $U --lengths 32k,128k,256k --depths 10,50,90 --label C262 --out $N | python3 -c "import sys,json; [print(r['length'],r['depth'],r.get('found'),r.get('distractor_given'),r.get('seconds')) for r in map(json.loads,sys.stdin)]"
echo "== A5 quality (20 prompts)"; python3 $S/quality.py --url $U/v1 --model $M --label C262 --out $R/strata-canonical-quality-2026-10-07.jsonl | tail -1
echo "== A6 GSM8K"; python3 $S/gsm8k.py --url $U/v1 --model $M --label C262 --out $R/strata-canonical-gsm8k-2026-10-07.jsonl --workers 2 | tail -1
echo A-DONE
