#!/usr/bin/env bash
# E6 来源域手术 · 自动续跑器
# 本机环境对单进程持续时间有限制（约 180s 会被回收），因此把工作切成多轮，
# 每轮限量移动；脚本幂等，可随时中断重跑。结束条件：所有清单清零（REMAINING=0）。
cd "$(dirname "$0")/.." || exit 1
LOG=data/_surgery_log.txt
PROG=data/_surgery_progress.txt
: > "$LOG"; : > "$PROG"
for i in $(seq 1 80); do
  ./.venv/Scripts/python.exe -u tools/apply_surgery_20261007.py --max-files 500 >> "$LOG" 2>&1
  rem=$(grep -o "REMAINING=[0-9]*" "$LOG" | tail -1 | cut -d= -f2)
  if [ -z "$rem" ]; then rem=UNKNOWN; fi
  echo "round $i: remaining_files=$rem" | tee -a "$PROG"
  if [ "$rem" = "0" ]; then
    echo "ALL_DONE" | tee -a "$PROG"
    exit 0
  fi
done
echo "MAX_ROUNDS_REACHED" | tee -a "$PROG"
