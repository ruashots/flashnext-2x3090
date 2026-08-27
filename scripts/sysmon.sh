#!/usr/bin/env bash
# Sample GPU memory, host memory and NVMe read throughput once per second.
# Usage: sysmon.sh <outfile> <seconds>
OUT="${1:-/tmp/sysmon.csv}"
DUR="${2:-120}"
echo "ts,gpu0_mib,gpu1_mib,gpu0_util,gpu1_util,mem_used_mb,mem_cache_mb,mem_avail_mb,nvme_read_mbs" > "$OUT"
prev=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats)
for ((i=0;i<DUR;i++)); do
  sleep 1
  mapfile -t G < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  g0m=${G[0]%%,*}; g0u=${G[0]##*,}
  g1m=${G[1]%%,*}; g1u=${G[1]##*,}
  read -r used cache avail < <(free -m | awk '/^Mem:/{print $3,$6,$7}')
  cur=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats)
  rd=$(( (cur - prev) * 512 / 1048576 ))   # diskstats field 6 = sectors read, 512 B each
  prev=$cur
  echo "$(date +%s),$g0m,$g1m,$g0u,$g1u,$used,$cache,$avail,$rd" >> "$OUT"
done
