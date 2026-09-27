#!/usr/bin/env bash
# One-booking, no-retune eager then graph pTimer comparison. No cleanup/poweroff.
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
if [[ ! -f "$source_dir/ptimer_standalone_replay.py" ]]; then
  printf 'standalone source is missing\n' >&2
  exit 2
fi
check_source() {
  local expected=$1
  local filename=$2
  local actual
  actual=$(sha256sum -- "$source_dir/$filename")
  actual=${actual%% *}
  if [[ "$actual" != "$expected" ]]; then
    printf 'source digest mismatch: %s\n' "$filename" >&2
    exit 2
  fi
}
check_source ca6f2b7c073266f306a7e87e3df90d31a97598fef18ee78b9aaf3552dd2ce0c8 ptimer_standalone_replay.py
check_source 3264be8616436aedf522991557f9515dc675a58b55649c8aeb7993417fadce99 ptimer_standalone_score.py
check_source 5dd5e7ac7cfcf0bc964d00869a115ca477fe43dcc200691082c3155a77c9a76f ptimer_posthoc_audit.py
check_source 8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd ptimer_trace.py
check_source 2770b3ad3157fee56409da1416b09101ff1c3c4ddfeb5fd6913865fabef8ded8 inflight_trace_v2.py
export PATH="$venv_dir/bin:$PATH"
command -v ninja >/dev/null || {
  printf 'ninja unavailable on selected PATH\n' >&2
  exit 2
}
expected_plugin=${LLR_EXPECTED_PLUGIN_SHA256:-ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b}
if [[ ! "$expected_plugin" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'expected plugin digest must be lowercase SHA-256\n' >&2
  exit 2
fi
actual_plugin=$(sha256sum -- "$plugin_so")
actual_plugin=${actual_plugin%% *}
if [[ "$actual_plugin" != "$expected_plugin" ]]; then
  printf 'plugin digest differs; review new build identity before running cells\n' >&2
  exit 2
fi
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"

cell_run() {
  local mode=$1
  local cell_dir=$2
  mkdir -m 700 -- "$cell_dir/inspector"
  local cycles=1000000
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
      --nproc-per-node=2 "$source_dir/ptimer_standalone_replay.py" \
      --mode "$mode" --private-dir "$cell_dir" --sleep-cycles "$cycles" \
      >"$cell_dir/runner.stdout" 2>"$cell_dir/runner.stderr"
  local digest_pair before_sha after_sha
  digest_pair=$("$venv_dir/bin/python" -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")
        if line.startswith("{")]
rows = [row for row in rows if row.get("schema") == "tp-ptimer-standalone-v1"]
if len(rows) != 1 or rows[0]["numeric_result"] != "pass":
    raise SystemExit("missing or duplicate standalone result")
print(rows[0]["before_sha256"], rows[0]["after_sha256"])
' "$cell_dir/runner.stdout")
  read -r before_sha after_sha <<<"$digest_pair"
  "$venv_dir/bin/python" "$source_dir/ptimer_standalone_score.py" \
    --private-dir "$cell_dir" --mode "$mode" \
    --before-sha256 "$before_sha" --after-sha256 "$after_sha" \
    >"$cell_dir/score.json"
  "$venv_dir/bin/python" -c '
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"mode": row["mode"], "outcome": row["outcome"]}, sort_keys=True))
' "$cell_dir/score.json"
}

eager_dir=$(mktemp -d -p "$private_base" ptimer-eager.XXXXXXXX)
cell_run eager "$eager_dir"
eager_outcome=$("$venv_dir/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["outcome"])' \
  "$eager_dir/score.json")
if [[ "$eager_outcome" != eager_instrument_pass ]]; then
  printf 'eager instrument did not pass; graph cell not run\n' >&2
  exit 1
fi

graph_dir=$(mktemp -d -p "$private_base" ptimer-graph.XXXXXXXX)
cell_run graph "$graph_dir"
printf 'Both cells finished. Preserve private directories; no files were deleted.\n'
