#!/usr/bin/env bash
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir=/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_offline800_8rounds_v1
mkdir -p "$run_dir/logs"
bash "$script_dir/run.sh" run >> "$run_dir/logs/pipeline.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/logs/launcher_exit_code.txt"
exit "$result"
