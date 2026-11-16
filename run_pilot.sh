#!/usr/bin/env bash
# The only command the user runs. Creates/reuses .venv (Python 3.12, uv), installs torch from
# the CUDA 13.0 index, downloads models/SAEs/data, runs every stage and writes the report.
#
#   bash run_pilot.sh                 full pilot (runs the Step 0 check first if needed)
#   bash run_pilot.sh --check         Step 0 only -> runs/check/decision.json
#   bash run_pilot.sh --smoke         tiny end-to-end run + acceptance checks (< 10 min on the GB10)
#   bash run_pilot.sh --resume        continue the latest run (skips stages with a DONE marker)
#   bash run_pilot.sh --out DIR       choose the run directory
#   bash run_pilot.sh --check --force rerun the check after config/check.yaml changed (recorded)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECK=0; SMOKE=0; RESUME=0; FORCE=0; DEV=0; OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --smoke) SMOKE=1 ;;
    --resume) RESUME=1 ;;
    --force) FORCE=1 ;;
    --dev-tiny) DEV=1 ;;                     # developer plumbing test (tiny random model)
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    --out=*) OUT="${1#--out=}" ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

log() { echo "[run_pilot $(date +%H:%M:%S)] $*"; }

# ------------------------------------------------------------------ environment for Python
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas      # Triton (incl. Gated DeltaNet kernels) on sm_121
[[ -x "$TRITON_PTXAS_PATH" ]] || log "note: $TRITON_PTXAS_PATH not found (needed only for Triton kernels on the GB10)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ GPU / torch index
HAVE_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then HAVE_GPU=1; fi
if [[ $HAVE_GPU -eq 1 ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu130"
  [[ "$(uname -m)" == "aarch64" ]] || log "note: $(uname -m) host; the target is the aarch64 GB10"
elif [[ $SMOKE -eq 1 || $DEV -eq 1 ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  log "WARNING: no NVIDIA GPU visible. Using CPU torch: the CPU path exists ONLY for the smoke check."
else
  log "ERROR: no NVIDIA GPU visible (nvidia-smi failed). The pilot needs the GB10; only --smoke runs on CPU."
  exit 1
fi

# ------------------------------------------------------------------ uv + venv
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  python3 -m pip install --user --quiet uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x .venv/bin/python ]]; then
  log "creating .venv (Python 3.12)"
  uv venv --python 3.12 .venv
fi
PY="$ROOT/.venv/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { log "ERROR: .venv is not Python 3.12; remove .venv and rerun"; exit 1; }

stamp_of() { cat "$@" | sha256sum | cut -c1-16; }
CHECK_STAMP=".venv/.stamp-check-$(echo "$TORCH_INDEX" | sha256sum | cut -c1-8)-$(stamp_of requirements-check.txt)"
FULL_STAMP=".venv/.stamp-full-$(echo "$TORCH_INDEX" | sha256sum | cut -c1-8)-$(stamp_of requirements.txt requirements-check.txt)"
NEED_FULL=1; [[ $CHECK -eq 1 ]] && NEED_FULL=0

PREFETCH_PID=""
if [[ ! -f "$CHECK_STAMP" || ( $NEED_FULL -eq 1 && ! -f "$FULL_STAMP" ) ]]; then
  log "installing dependencies with uv (torch from $TORCH_INDEX)"
  uv pip install --python "$PY" --quiet "huggingface_hub==2.4.0"
  # start the model download now, overlapped with the PyTorch install
  KIND=pilot; [[ $CHECK -eq 1 ]] && KIND=check; [[ $SMOKE -eq 1 ]] && KIND=smoke
  if [[ $DEV -eq 0 ]]; then
    mkdir -p runs
    "$PY" -m pilot.prefetch "$KIND" > runs/.prefetch.log 2>&1 &
    PREFETCH_PID=$!
    log "model prefetch started in the background (log: runs/.prefetch.log)"
  fi
  uv pip install --python "$PY" --quiet --index-url "$TORCH_INDEX" torch
  uv pip install --python "$PY" --quiet -r requirements-check.txt
  touch "$CHECK_STAMP"
  if [[ $NEED_FULL -eq 1 ]]; then
    uv pip install --python "$PY" --quiet -r requirements.txt
    touch "$FULL_STAMP"
  fi
fi
"$PY" - <<'EOF' || exit 1
import torch, sys
v = torch.__version__
print(f"[run_pilot] torch {v} (CUDA build {torch.version.cuda}), cuda available: {torch.cuda.is_available()}")
EOF
if [[ -n "$PREFETCH_PID" ]]; then
  log "waiting for the model prefetch to finish"
  wait "$PREFETCH_PID" || log "prefetch failed (see runs/.prefetch.log); the stages will retry the download"
fi

# ------------------------------------------------------------------ memory safety (unified memory)
MEM_TOTAL_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
if [[ $MEM_TOTAL_GB -ge 116 ]]; then MEMMAX=100G; else MEMMAX="$(( MEM_TOTAL_GB * 85 / 100 ))G"; fi
PREFIX=()
if command -v systemd-run >/dev/null 2>&1 && \
   systemd-run --user --scope --quiet -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 true >/dev/null 2>&1; then
  PREFIX=(systemd-run --user --scope --quiet -p MemoryMax="$MEMMAX" -p MemorySwapMax=0)
  export PILOT_SYSTEMD_SCOPE=1 PILOT_SYSTEMD_NOTE="MemoryMax=$MEMMAX MemorySwapMax=0"
  log "running inside a systemd scope: MemoryMax=$MEMMAX, MemorySwapMax=0"
else
  export PILOT_SYSTEMD_SCOPE=0 PILOT_SYSTEMD_NOTE="systemd-run --user --scope unavailable"
  log "WARNING: systemd-run --user --scope is unavailable here; continuing without a MemoryMax cap" \
      "(the in-process MemAvailable watchdog still runs). Recommended once: sudo swapoff -a"
fi

run_py() { "${PREFIX[@]}" "$PY" -m pilot.cli "$@"; }

# ------------------------------------------------------------------ dispatch
ARGS=()
[[ $RESUME -eq 1 ]] && ARGS+=(--resume)
[[ -n "$OUT" ]] && ARGS+=(--out "$OUT")

if [[ $CHECK -eq 1 ]]; then
  CARGS=(--check); [[ $SMOKE -eq 1 ]] && CARGS+=(--smoke); [[ $FORCE -eq 1 ]] && CARGS+=(--force); [[ $DEV -eq 1 ]] && CARGS+=(--dev-tiny)
  run_py "${CARGS[@]}"
  exit $?
fi

if [[ $SMOKE -eq 1 || $DEV -eq 1 ]]; then
  KIND=smoke; FLAG=--smoke
  if [[ $DEV -eq 1 ]]; then KIND=dev; FLAG=--dev-tiny; fi
  T0=$(date +%s)
  set +e
  run_py "$FLAG" "${ARGS[@]}"
  RC=$?
  set -e
  RUN_DIR="$(cat "runs/.last_$KIND" 2>/dev/null || true)"
  log "$KIND run finished with exit code $RC in $(( $(date +%s) - T0 )) s: $RUN_DIR"
  [[ $RC -eq 0 && -n "$RUN_DIR" ]] || exit $RC
  log "acceptance checks (pytest, identity, SAE sanity, NaN/Inf, outputs, --resume < 30 s, --check --smoke)"
  VARGS=(--verify-smoke --out "$RUN_DIR"); [[ $DEV -eq 1 ]] && VARGS+=(--dev-tiny)
  run_py "${VARGS[@]}"
  exit $?
fi

run_py "${ARGS[@]}"
