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
#
# v2 (pilot-v2-amendment.md); runs land in runs/v2/<phase>/<run_id>/:
#   bash run_pilot.sh --v2 --prefetch        download every model, SAE file and dataset the full run needs
#   bash run_pilot.sh --v2 --smoke           prefetch, then every stage on tiny data + checks + projections
#   bash run_pilot.sh --v2 --dev             development run (Qwen3-1.7B truth + refusal, 50% depth)
#   bash run_pilot.sh --v2 --render-prereg   regenerate prereg/PREREGISTRATION.md from prereg/hypotheses.yaml
#   bash run_pilot.sh --v2 --freeze          hash, commit and tag the preregistration
#   bash run_pilot.sh --v2 --confirm [--deviation "<reason>"]   confirmatory run (needs a valid freeze)
#   add --resume (or --out DIR) to continue a v2 run
#
# v3 (documents/pilot-v3-plan.md); runs land in runs/v3/<phase>/<run_id>/; needs HF_TOKEN in the environment (Gemma):
#   bash run_pilot.sh --v3 --prefetch | --smoke | --dev | --render-prereg | --freeze | --confirm [--deviation "<reason>"]
#   (one-command pipeline: bash run_v3.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECK=0; SMOKE=0; RESUME=0; FORCE=0; DEV=0; OUT=""
V2=0; V3=0; V2MODE=""; DEVIATION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --v2) V2=1 ;;
    --v3) V3=1 ;;
    --prefetch|--dev|--freeze|--confirm|--render-prereg) V2MODE="${1#--}" ;;
    --deviation) DEVIATION="${2:?--deviation needs a reason}"; shift ;;
    --deviation=*) DEVIATION="${1#--deviation=}" ;;
    --check) CHECK=1 ;;
    --smoke) SMOKE=1 ;;
    --resume) RESUME=1 ;;
    --force) FORCE=1 ;;
    --dev-tiny) DEV=1 ;;                     # developer plumbing test (tiny random model)
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    --out=*) OUT="${1#--out=}" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

log() { echo "[run_pilot $(date +%H:%M:%S)] $*"; }

[[ $V2 -eq 1 && $V3 -eq 1 ]] && { echo "use either --v2 or --v3"; exit 2; }
if [[ $V2 -eq 1 || $V3 -eq 1 ]]; then
  [[ $SMOKE -eq 1 ]] && V2MODE="smoke"
  [[ -n "$V2MODE" ]] || { echo "--v2 / --v3 need one of --prefetch --smoke --dev --render-prereg --freeze --confirm"; exit 2; }
  [[ $CHECK -eq 0 ]] || { echo "--check is a v1 mode; do not combine it with --v2"; exit 2; }
  [[ -z "$DEVIATION" || "$V2MODE" == "confirm" ]] || { echo "--deviation applies to --v2 --confirm only"; exit 2; }
elif [[ -n "$V2MODE" || -n "$DEVIATION" ]]; then
  echo "--$V2MODE / --deviation are v2 / v3 modes: add --v2 or --v3"; exit 2
fi
CPU_OK=0
[[ $SMOKE -eq 1 || $DEV -eq 1 ]] && CPU_OK=1
[[ ( $V2 -eq 1 || $V3 -eq 1 ) && "$V2MODE" =~ ^(prefetch|freeze|render-prereg)$ ]] && CPU_OK=1

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
elif [[ $CPU_OK -eq 1 ]]; then
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
  if [[ $DEV -eq 0 && $V2 -eq 0 && $V3 -eq 0 ]]; then
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

# v2 only: Triton Gated DeltaNet kernels for Qwen3.5-4B (pure Python + Triton, no CUDA 12 build). Best effort;
# the model falls back to the pure-PyTorch implementation without them. PILOT_V2_FLA=0 skips this.
FLA_STAMP=".venv/.stamp-fla-0.6.0"
if [[ $V2 -eq 1 && $HAVE_GPU -eq 1 && "${PILOT_V2_FLA:-1}" != "0" && ! -f "$FLA_STAMP" ]]; then
  log "installing flash-linear-attention 0.6.0 (Triton kernels for the Qwen3.5 Gated DeltaNet layers; best effort)"
  if uv pip install --python "$PY" --quiet "flash-linear-attention==0.6.0" "fla-core==0.6.0" einops; then
    touch "$FLA_STAMP"
  else
    log "WARNING: flash-linear-attention did not install; Qwen3.5-4B will use the slower PyTorch fallback"
  fi
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
run_v2() { "${PREFIX[@]}" "$PY" -m pilot.cli_v2 "$@"; }
run_v3() { "${PREFIX[@]}" "$PY" -m pilot.cli_v3 "$@"; }

# ------------------------------------------------------------------ v3 dispatch
if [[ $V3 -eq 1 ]]; then
  VARGS=("--$V2MODE")
  [[ $RESUME -eq 1 ]] && VARGS+=(--resume)
  [[ -n "$OUT" ]] && VARGS+=(--out "$OUT")
  [[ $DEV -eq 1 ]] && VARGS+=(--dev-tiny)
  [[ -n "$DEVIATION" ]] && VARGS+=(--deviation "$DEVIATION")
  if [[ $DEV -eq 0 && "$V2MODE" =~ ^(prefetch|smoke|dev|confirm)$ && -z "${HF_TOKEN:-}" ]]; then
    log "ERROR: HF_TOKEN is not set. google/gemma-2-2b is gated: accept its license on huggingface.co, then"
    log "       export HF_TOKEN=hf_... (the token is read from the environment only and never stored)."
    exit 3
  fi
  if [[ "$V2MODE" != "smoke" ]]; then
    run_v3 "${VARGS[@]}"
    exit $?
  fi
  T0=$(date +%s)
  set +e
  run_v3 "${VARGS[@]}"
  RC=$?
  set -e
  RUN_DIR="$(cat runs/v3/smoke/.last 2>/dev/null || true)"
  log "v3 smoke run finished with exit code $RC in $(( $(date +%s) - T0 )) s: $RUN_DIR"
  [[ $RC -eq 0 && -n "$RUN_DIR" ]] || exit $RC
  log "v3 acceptance checks (pytest, SAE fidelity + scaling + layer index, prefix cache, identity, zero cost, freeze enforcement, prefetch, projection)"
  CARGS=(--verify-smoke --out "$RUN_DIR"); [[ $DEV -eq 1 ]] && CARGS+=(--dev-tiny)
  run_v3 "${CARGS[@]}"
  exit $?
fi

# ------------------------------------------------------------------ v2 dispatch
if [[ $V2 -eq 1 ]]; then
  VARGS=("--$V2MODE")
  [[ $RESUME -eq 1 ]] && VARGS+=(--resume)
  [[ -n "$OUT" ]] && VARGS+=(--out "$OUT")
  [[ $DEV -eq 1 ]] && VARGS+=(--dev-tiny)
  [[ -n "$DEVIATION" ]] && VARGS+=(--deviation "$DEVIATION")
  if [[ "$V2MODE" != "smoke" ]]; then
    run_v2 "${VARGS[@]}"
    exit $?
  fi
  T0=$(date +%s)
  set +e
  run_v2 "${VARGS[@]}"
  RC=$?
  set -e
  RUN_DIR="$(cat runs/v2/smoke/.last 2>/dev/null || true)"
  log "v2 smoke run finished with exit code $RC in $(( $(date +%s) - T0 )) s: $RUN_DIR"
  [[ $RC -eq 0 && -n "$RUN_DIR" ]] || exit $RC
  log "v2 acceptance checks (pytest, identity incl. decoding, zero cost, extension, freeze enforcement, prefetch, projection)"
  CARGS=(--verify-smoke --out "$RUN_DIR"); [[ $DEV -eq 1 ]] && CARGS+=(--dev-tiny)
  run_v2 "${CARGS[@]}"
  exit $?
fi

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
