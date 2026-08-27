#!/usr/bin/env bash
# Launch llama-server with the expert layers named by <band> held on the host.
# The two bands are chosen one inside each card's layer range, so the bytes end up
# balanced without touching -ts, which would also move the KV cache and dense layers.
# Usage: run_ot.sh <band_regex> <ctx> <quant_dir> <tag> [extra flags...]
set -uo pipefail
BAND="${1:?band}"; CTX="${2:?ctx}"; QDIR="${3:-UD-IQ4_XS}"; TAG="${4:?tag}"; shift 4 || true
MODEL=$(ls /mnt/models/flashnext/$QDIR/*-00001-of-*.gguf)
BIN=/opt/llama.cpp/build/bin/llama-server
LOGDIR=/mnt/models/flashnext-bench/runs; mkdir -p "$LOGDIR"
LOG="$LOGDIR/server-$TAG.log"

pkill -f 'llama-serve[r]' 2>/dev/null; sleep 4
echo "### $TAG band=$BAND ctx=$CTX quant=$QDIR $(date -Is)" | tee "$LOG"

LD_LIBRARY_PATH=/opt/llama.cpp/build/bin setsid nohup "$BIN" \
  -m "$MODEL" --host 0.0.0.0 --port 8080 --alias flashnext \
  -ngl 99 -sm layer -fit off -c "$CTX" -fa on -ctk f16 -ctv f16 \
  -np 1 --cache-ram 0 -b 2048 -ub 512 -t 8 --threads-batch 8 \
  --jinja --reasoning-preserve --no-warmup -lv 4 \
  -ot "^per_layer_token_embd\.weight$=CPU" \
  -ot "${BAND}=CPU" \
  "$@" >> "$LOG" 2>&1 < /dev/null &

SPID=$!
for i in $(seq 1 240); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; grep -iE "error|out of memory|failed|assert" "$LOG" | tail -8; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/health || true)" = "200" ] && {
      echo "HEALTHY after $((i*5))s"
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
      free -m | awk '/^Mem:/{printf "RAM used %s MB, cache %s MB, avail %s MB\n",$3,$6,$7}'
      grep -iE "CPU_Mapped|model buffer size|indexer KV|KV self|kv cache" "$LOG" | tail -8
      exit 0; }
  sleep 5
done
echo "TIMEOUT"; tail -20 "$LOG"; exit 2
