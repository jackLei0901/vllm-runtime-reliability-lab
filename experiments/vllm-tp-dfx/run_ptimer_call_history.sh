#!/usr/bin/env bash
# Frozen three-cell closeout. No cleanup or poweroff; raw files stay private.
set -euo pipefail
umask 077

if (( $# != 4 )); then
  printf 'usage: %s VENV_DIR PLUGIN_SO PRIVATE_BASE LAB_ROOT\n' "$0" >&2
  exit 2
fi
venv_dir=$(realpath -e -- "$1")
plugin_so=$(realpath -e -- "$2")
private_base=$(realpath -e -- "$3")
lab_root=$(realpath -e -- "$4")
source_dir="$lab_root/experiments/vllm-tp-dfx"
if [[ ! -x "$venv_dir/bin/python" || ! -x "$venv_dir/bin/torchrun" ]]; then
  printf 'selected venv lacks python or torchrun\n' >&2
  exit 2
fi
export PATH="$venv_dir/bin:$PATH"
command -v ninja >/dev/null || {
  printf 'ninja unavailable on selected PATH\n' >&2
  exit 2
}
check_source() {
  local expected=$1 filename=$2 actual
  actual=$(sha256sum -- "$source_dir/$filename")
  actual=${actual%% *}
  if [[ "$actual" != "$expected" ]]; then
    printf 'source digest mismatch: %s\n' "$filename" >&2
    exit 2
  fi
}
check_source 61d939062274819f8f41ad330563ac45931f343524695190333196987d6d44d3 ptimer_call_history.py
check_source f869729baf414b58545a1d7db435a525922498897addb1cce2674d08ad3bb59f ptimer_call_history_score.py
check_source ca6f2b7c073266f306a7e87e3df90d31a97598fef18ee78b9aaf3552dd2ce0c8 ptimer_standalone_replay.py
check_source 3264be8616436aedf522991557f9515dc675a58b55649c8aeb7993417fadce99 ptimer_standalone_score.py
check_source 5dd5e7ac7cfcf0bc964d00869a115ca477fe43dcc200691082c3155a77c9a76f ptimer_posthoc_audit.py
check_source 8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd ptimer_trace.py
check_source 2770b3ad3157fee56409da1416b09101ff1c3c4ddfeb5fd6913865fabef8ded8 inflight_trace_v2.py
expected_plugin=${LLR_EXPECTED_PLUGIN_SHA256:-ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b}
if [[ ! "$expected_plugin" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'expected plugin digest must be lowercase SHA-256\n' >&2
  exit 2
fi
actual_plugin=$(sha256sum -- "$plugin_so")
actual_plugin=${actual_plugin%% *}
if [[ "$actual_plugin" != "$expected_plugin" ]]; then
  printf 'Inspector binary digest differs; stop before cells\n' >&2
  exit 2
fi
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"

for variant in no_eager same_comm other_comm; do
  cell_dir=$(mktemp -d -p "$private_base" "ptimer-${variant}.XXXXXXXX")
  mkdir -m 700 -- "$cell_dir/inspector"
  env \
    NCCL_PROFILER_PLUGIN="$plugin_so" \
    NCCL_DEBUG=TRACE \
    NCCL_DEBUG_SUBSYS=INIT,PROFILE \
    NCCL_DEBUG_FILE="$cell_dir/nccl.%p.log" \
    NCCL_INSPECTOR_DUMP_DIR="$cell_dir/inspector" \
    NCCL_INSPECTOR_ENABLE=1 \
    NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 \
    NCCL_INSPECTOR_DUMP_VERBOSE=1 \
    timeout 180s "$venv_dir/bin/torchrun" --standalone --nnodes=1 \
      --nproc-per-node=2 "$source_dir/ptimer_call_history.py" \
      --variant "$variant" --private-dir "$cell_dir" --sleep-cycles 1000000 \
      >"$cell_dir/runner.stdout" 2>"$cell_dir/runner.stderr"
  digest_triple=$("$venv_dir/bin/python" -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")
        if line.startswith("{")]
rows = [row for row in rows if row.get("schema") == "tp-ptimer-call-history-v1"]
if len(rows) != 1 or rows[0]["variant"] != sys.argv[2] or rows[0]["numeric_result"] != "pass":
    raise SystemExit("missing or duplicate call-history result")
print(rows[0]["before_sha256"], rows[0]["during_sha256"], rows[0]["after_sha256"])
' "$cell_dir/runner.stdout" "$variant")
  read -r before_sha during_sha after_sha <<<"$digest_triple"
  "$venv_dir/bin/python" "$source_dir/ptimer_call_history_score.py" \
    --private-dir "$cell_dir" --variant "$variant" \
    --before-sha256 "$before_sha" --during-sha256 "$during_sha" \
    --after-sha256 "$after_sha" >"$cell_dir/score.json"
  "$venv_dir/bin/python" -c '
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"variant": row["variant"], "outcome": row["outcome"]}, sort_keys=True))
' "$cell_dir/score.json"
done
printf 'Three cells finished. Preserve private directories; no files were deleted.\n'
