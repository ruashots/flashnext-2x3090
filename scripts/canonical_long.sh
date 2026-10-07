#!/bin/bash
# canonical_long.sh 512k|1m: with the server on the stretched setup (yarn, one request at a time), run the long speed
# ladder and the needle tests. Label prefix C512 / C1M.
SET=${1:?usage: canonical_long.sh 512k|1m}; S=$(cd "$(dirname "$0")" && pwd); R=${OUT:-$S/../raw}; T=$(mktemp -d); M=${MODEL:-Qwen3.8-Flash-Next-OrcaRouter}; U=${URL:-http://127.0.0.1:8080}
STRATA=${STRATA:?set STRATA to a plain clone of Strata at v0.1.40.2}
B=$R/strata-canonical-bench-2026-10-07.jsonl; N=$R/strata-canonical-needle-2026-10-07.jsonl
if [ "$SET" = 512k ]; then L=C512; CTX=300000,400000,500000; NL=300k,400k,500k; ND=10,50,90; MK=400k,500k; MD=50
else L=C1M; CTX=600000,900000,1000000; NL=1000k; ND=10,50,90; MK=1000k; MD=50; fi
# The server must already run strata-orca-512k.json or strata-orca-1m.json (setups/2026-10-strata-v0.1.40.2/).
curl -s -m 5 $U/v1/models | grep -o '"n_ctx": [0-9]*' | head -1
python3 $S/bench.py --url $U/v1 --model $M --label prewarm --out $T/prewarm-c.jsonl --contexts 60000 --tasks reason --runs 1 --warmup 0 >/dev/null 2>&1
echo "== speed"; python3 $S/bench.py --url $U/v1 --model $M --label $L --out $B --contexts 0,4000 --tasks code,reason --runs 3 2>&1 | grep ">>"
python3 $S/bench.py --url $U/v1 --model $M --label $L --out $B --contexts $CTX --tasks reason --runs 1 --warmup 0 --timeout 5400 2>&1 | grep ">>"
echo "== needle"; python3 $STRATA/tools/needle_bench.py --url $U --lengths $NL --depths $ND --timeout 5400 --out $T/needle-$SET.json
[ "$SET" = 1m ] && python3 $STRATA/tools/needle_bench.py --url $U --lengths 600k,800k --depths 50 --timeout 5400 --out $T/needle-$SET-b.json
for f in $T/needle-$SET.json $T/needle-$SET-b.json; do [ -f $f ] && python3 -c "import json,sys; [print(json.dumps({'label':'$L','test':'needle',**r})) for r in json.load(open('$f'))]" >> $N; done
echo "== multikey"; python3 $S/needle_multikey.py --strata $STRATA --url $U --lengths $MK --depths $MD --label $L --timeout 5400 --out $N | python3 -c "import sys,json; [print(r['length'],r['depth'],r.get('found'),r.get('distractor_given'),r.get('seconds')) for r in map(json.loads,sys.stdin)]"
echo "== quality"; python3 $S/quality.py --url $U/v1 --model $M --label $L --out $R/strata-canonical-quality-2026-10-07.jsonl | tail -1
echo "== gsm8k first 250"; python3 $S/gsm8k.py --url $U/v1 --model $M --label $L --out $R/strata-canonical-gsm8k-2026-10-07.jsonl --limit 250 | tail -1
echo LONG-DONE-$SET
