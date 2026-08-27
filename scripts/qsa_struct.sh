#!/usr/bin/env bash
# Structural check: is Qwen Sparse Attention actually engaged, or does the branch
# fall back to dense? Read the graph rather than timing it. qwen4exp.cpp builds
# top_k with width = min(n_kv, indexer_top_k + compress_ratio - 1) = min(n_kv, 2051).
# If that tensor appears with extent 2051 while n_kv is 16384, QSA is engaged.
set -uo pipefail
BAND="${1:?band}"; UB="${2:?ub}"
B=/mnt/models/flashnext-bench
LOG=$B/sched-debug.log
MODEL=$(ls /mnt/models/flashnext/UD-IQ4_XS/*-00001-of-*.gguf)

pkill -f 'llama-serve[r]' 2>/dev/null; sleep 4
LD_LIBRARY_PATH=/opt/llama.cpp/build/bin GGML_SCHED_DEBUG=2 setsid nohup \
  /opt/llama.cpp/build/bin/llama-server -m "$MODEL" \
  --host 0.0.0.0 --port 8080 --alias flashnext \
  -ngl 99 -sm layer -fit off -c 73728 -fa on -ctk f16 -ctv f16 \
  -np 1 --cache-ram 0 -b 2048 -ub "$UB" -t 8 --threads-batch 8 \
  --jinja --no-warmup -lv 6 \
  -ot "^per_layer_token_embd\.weight$=CPU" -ot "${BAND}=CPU" \
  > "$LOG" 2>&1 < /dev/null &

for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/health || true)" = "200" ] && break
  sleep 5
done
echo "server up, sending a 16k prompt"
python3 - <<'PY'
import json,urllib.request,random
random.seed(11)
w="harbor lantern quarry meadow cinder ledger thicket furnace pebble ravine compass timber".split()
body=" ".join(random.choice(w) for _ in range(14000))
req=urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",method="POST",
  data=json.dumps({"messages":[{"role":"user","content":body+"\n\nReply with the single word OK."}],
  "max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode(),
  headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=600))
print("prompt_tokens:", r["usage"]["prompt_tokens"])
PY
sleep 3
echo "=== indexer_top_k tensors in the graph dump ==="
grep -iE "indexer_top_k|indexer" "$LOG" | head -20
echo "=== n_kv / kv cache lines ==="
grep -iE "n_kv|indexer KV|kv cache|kv self" "$LOG" | head -10
echo "=== graph split count ==="
grep -c "split" "$LOG" || true
pkill -f 'llama-serve[r]' 2>/dev/null
