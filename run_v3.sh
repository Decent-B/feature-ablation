#!/usr/bin/env bash
# run_v3.sh: the whole v3 pilot in one command (documents/pilot-v3-plan.md).
#
#   export HF_TOKEN=hf_...        once per shell: google/gemma-2-2b is gated (accept its license on huggingface.co)
#   bash run_v3.sh                start; it continues in the background, so you can close the terminal
#   bash run_v3.sh --status       where it is, and the summary once it has finished or stopped
#   bash run_v3.sh --stop         stop it (finished work is kept)
#   bash run_v3.sh --foreground   like the default, but stays attached to this terminal
#
# Steps (about 2.5 h on the GB10 after the downloads):
#   1. download models, SAEs and datasets (~23 GB; up to 3 tries, for network blips)
#   2. smoke test (~20 min; measures the SAE input scaling and writes config/v3_sae_input_norm.yaml)
#   3. development run (~10 min; harness health only, no results)
#   4. finalise the preregistration (status: final), render, commit, freeze (tag prereg-v3-frozen)
#   5. confirmatory run (~2 h; the time guard skips the lowest-priority blocks if time runs short)
#
# It stops at the first error and says why (a failed smoke check counts as an error). Running it again
# continues from where it stopped: finished steps are skipped and an interrupted step is resumed.
# The token is read from the environment only; it is never written to a file or a log.
#
# Log: runs/v3/pipeline/run_v3.log    Summary: runs/v3/pipeline/SUMMARY.txt

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1

STATE=runs/v3/pipeline
LOG=$STATE/run_v3.log
LOCK=$STATE/lock
SUMMARY=$STATE/SUMMARY.txt
TAG=prereg-v3-frozen
mkdir -p "$STATE"

# developer plumbing test only: tiny random models on CPU (never use for real results)
TINY=()
[[ "${RUN_V3_DEV_TINY:-0}" == 1 ]] && TINY=(--dev-tiny)

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }
mark() { echo "$(ts) ${2:-}" > "$STATE/$1"; }
has() { [[ -f "$STATE/$1" ]]; }
is_running() { ! flock -n "$LOCK" true 2>/dev/null; }
stop_requested() { [[ -f "$STATE/stop.request" ]]; }
last_run() { [[ -f "runs/v3/$1/.last" ]] && tr -d '\n' < "runs/v3/$1/.last"; }
frozen() { git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; }
pilot() { bash run_pilot.sh "$@"; }
AGAIN="After fixing the cause, run 'bash run_v3.sh' again: it continues from this step."

preflight() {
  local bad=0
  if [[ ${#TINY[@]} -eq 0 && -z "${HF_TOKEN:-}" ]]; then
    echo "STOP: HF_TOKEN is not set. google/gemma-2-2b is gated: accept its license on huggingface.co with your"
    echo "      account, create a read token, then: export HF_TOKEN=hf_...   (never stored by this script)"; bad=1
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "STOP: $PWD is not a git repository (the freeze needs git)."; bad=1
  elif [[ -z "$(git config user.name)" || -z "$(git config user.email)" ]]; then
    echo "STOP: git has no user.name / user.email on this machine (the freeze makes a commit). Set them once:"
    echo "  git config --global user.name \"Your Name\"; git config --global user.email \"you@example.com\""
    bad=1
  fi
  if [[ ${#TINY[@]} -eq 0 ]] && ! nvidia-smi -L >/dev/null 2>&1; then
    echo "STOP: no NVIDIA GPU visible (nvidia-smi failed)."; bad=1
  fi
  local hf=${HF_HOME:-$HOME/.cache/huggingface}
  mkdir -p "$hf"
  local free_gb=$(( $(df -Pk "$hf" | awk 'NR==2 {print $4}') / 1048576 ))
  if (( free_gb < 30 )); then
    echo "WARNING: only ${free_gb} GB free under $hf; the downloads need about 23 GB (fine if they are already there)."
  fi
  return $bad
}

summary() {
  local note=${1:-} sm dv cf
  sm=$(last_run smoke); dv=$(last_run dev); cf=$(last_run confirm)
  {
    echo "v3 summary ($(ts))"
    [[ -n "$note" ]] && echo "STATUS: $note"
    echo
    if [[ -n "$sm" && -f "$sm/smoke_report.md" ]]; then
      echo "Smoke test:    $(grep -o '[0-9]* checks passed, [0-9]* failed' "$sm/smoke_report.md" | head -1)"
      echo "               $sm/smoke_report.md"
    else
      echo "Smoke test:    no report"
    fi
    if [[ -n "$dv" && -f "$dv/dev_report.md" ]]; then
      echo "Development:   $dv/dev_report.md"
    else
      echo "Development:   no report"
    fi
    if [[ -n "$cf" && -f "$cf/report_v3.md" ]]; then
      echo "Confirmatory:  $(grep -m1 '^## Headline' "$cf/report_v3.md" | sed 's/^## //; s/\*//g')"
      echo "               $cf/report_v3.md (\"Deviations and skipped blocks\" lists any blocks skipped for time)"
    else
      echo "Confirmatory:  no report"
    fi
  } | tee "$SUMMARY"
}

halt() {   # stop the pipeline with a reason
  say "STOPPED: $1"
  summary "stopped: $1"
  return 1
}

run_step() {   # run_step <phase> <label> <run_pilot.sh args...>; stops the pipeline if the step fails
  local phase=$1 label=$2; shift 2
  local args=("$@" "${TINY[@]}") note="" rc dir detail
  if has "$phase.started"; then args+=(--resume); note=" (resuming)"; fi
  mark "$phase.started"
  say "$label$note"
  pilot "${args[@]}"; rc=$?
  if stop_requested; then halt "stopped on request during: $label. Run 'bash run_v3.sh' again to continue"; return 1; fi
  if (( rc != 0 )); then
    dir=$(last_run "$phase")
    detail="$dir/log.txt"
    [[ "$phase" == smoke && -f "$dir/smoke_report.md" ]] && detail="$dir/smoke_report.md"
    halt "$label failed (exit $rc). Details: $detail. $AGAIN"
    return 1
  fi
  mark "$phase.done"
}

main() {
  rm -f "$STATE/stop.request"
  trap ':' INT     # on --stop, wait for the current step to exit cleanly, then halt with a summary
  say "run_v3.sh started (pid $$) in $PWD${TINY:+ [dev-tiny plumbing test]}"
  local rc a

  # 1. downloads (retried: a network blip is not an error in the pipeline)
  if ! has prefetch.done; then
    rc=1
    for a in 1 2 3; do
      say "step 1/5: download models, SAEs and datasets (try $a of 3)"
      pilot --v3 --prefetch "${TINY[@]}"; rc=$?
      if stop_requested; then halt "stopped on request during the downloads. Run 'bash run_v3.sh' again to continue"; return 1; fi
      (( rc == 0 )) && break
      (( a < 3 )) && { say "download try $a failed (exit $rc); trying again in 2 min"; sleep 120; }
    done
    (( rc == 0 )) || { halt "the downloads failed 3 times (exit $rc; see above). $AGAIN"; return 1; }
    mark prefetch.done
  fi

  # 2. smoke test; 3. development run
  has smoke.done || run_step smoke "step 2/5: smoke test" --v3 --smoke || return 1
  has dev.done || run_step dev "step 3/5: development run" --v3 --dev || return 1

  # 4. finalise + freeze the preregistration
  if frozen; then
    say "step 4/5: preregistration already frozen (tag $TAG)"
  else
    say "step 4/5: finalise, render, commit and freeze the preregistration"
    sed -i 's/^status: draft/status: final/' prereg/v3/hypotheses.yaml
    pilot --v3 --render-prereg || { halt "rendering the preregistration failed (see above). $AGAIN"; return 1; }
    git add -u && git add -- run_v3.sh prereg/v3 config/v3.yaml config/v3_sae_input_norm.yaml src/pilot || { halt "git add failed (see above). $AGAIN"; return 1; }
    if ! git diff --cached --quiet; then
      git commit -q -m "v3: final preregistration" || { halt "git commit failed (see above). $AGAIN"; return 1; }
    fi
    pilot --v3 --freeze
    frozen || { halt "the freeze failed (see above). $AGAIN"; return 1; }
  fi
  mark freeze.done

  # 5. confirmatory run
  has confirm.done || run_step confirm "step 5/5: confirmatory run" --v3 --confirm || return 1

  say "all steps finished"
  summary "finished"
}

case "${1:-}" in
  "")
    exec 9> "$LOCK"     # held here and handed to the background process, so a second start is refused
    flock -n 9 || { echo "run_v3.sh is already running. Progress: bash run_v3.sh --status"; exit 1; }
    preflight || exit 1
    # setsid -f (not `&`): a background job started with `&` would ignore the stop signal
    setsid -f nohup bash "$0" --child >> "$LOG" 2>&1 < /dev/null
    echo "Started in the background. You can close this terminal."
    echo "  progress:  bash run_v3.sh --status     (or: tail -f $LOG)"
    echo "  stop:      bash run_v3.sh --stop"
    echo "  summary when it finishes or stops: $SUMMARY"
    ;;
  --child)              # internal: started by the default mode, with the lock already held on fd 9
    flock -n 9 || { say "another run_v3.sh is already running"; exit 1; }
    echo $$ > "$STATE/pid"
    main
    ;;
  --foreground)
    if is_running; then echo "run_v3.sh is already running. Progress: bash run_v3.sh --status"; exit 1; fi
    preflight || exit 1
    exec 9> "$LOCK"
    flock -n 9 || { echo "another run_v3.sh is already running"; exit 1; }
    echo $$ > "$STATE/pid"
    main 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
    ;;
  --status)
    if is_running; then echo "run_v3.sh is RUNNING (pid $(cat "$STATE/pid" 2>/dev/null))."; else echo "run_v3.sh is not running."; fi
    for s in prefetch smoke dev freeze confirm; do
      if has "$s.done"; then printf '  %-9s done      %s\n' "$s" "$(cat "$STATE/$s.done")"
      elif has "$s.started"; then printf '  %-9s started   %s\n' "$s" "$(cat "$STATE/$s.started")"
      else printf '  %-9s pending\n' "$s"; fi
    done
    if [[ -f "$SUMMARY" ]] && ! is_running; then echo; cat "$SUMMARY"; fi
    echo; echo "Last log lines ($LOG):"; tail -n 5 "$LOG" 2>/dev/null | cut -c1-200
    ;;
  --stop)
    if ! is_running; then echo "run_v3.sh is not running."; exit 0; fi
    touch "$STATE/stop.request"
    pid=$(cat "$STATE/pid")
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid"
    echo "Stop requested. Finished work is kept; run 'bash run_v3.sh' again to continue."
    ;;
  *)
    sed -n '2,7p' "$0"; exit 2
    ;;
esac
