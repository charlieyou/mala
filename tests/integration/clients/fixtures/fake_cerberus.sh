#!/usr/bin/env bash
set -euo pipefail

subcommand="${1:-}"
if [[ -z "$subcommand" ]]; then
  echo "missing subcommand" >&2
  exit 64
fi
shift

log_path="${CERBERUS_FAKE_LOG:?CERBERUS_FAKE_LOG must be set}"
state_root="${CERBERUS_STATE_ROOT:?CERBERUS_STATE_ROOT must be set}"
project_key="${CERBERUS_PROJECT_KEY:-mala-v2-smoke}"
run_key="${CERBERUS_RUN_KEY:-mala-${CLAUDE_SESSION_ID:-missing-session}}"
case_name="${CERBERUS_FAKE_CASE:?CERBERUS_FAKE_CASE must be set}"
run_dir="${state_root}/${project_key}/${run_key}"

printf '%s|%s|%s\n' "$(basename "$0")" "$subcommand" "$*" >> "$log_path"

write_output() {
  local round="$1"
  local reviewer="$2"
  local findings_json="$3"
  local reviewer_dir="${run_dir}/iterations/1/round-${round}/reviewers/${reviewer}"

  mkdir -p "$reviewer_dir"
  cat > "${reviewer_dir}/output.json" <<JSON
{
  "verdict": "fail",
  "summary": "fake ${reviewer} summary",
  "overall_confidence": 0.91,
  "strategy": "smoke",
  "round": ${round},
  "peer_responses_seen": 0,
  "findings": ${findings_json}
}
JSON
}

write_gate_state() {
  local verdict="$1"
  local exit_code="$2"

  cat > "${run_dir}/gate-state.json" <<JSON
{
  "schema_version": 2,
  "run_key": "${run_key}",
  "host": "generic",
  "project_key": "${project_key}",
  "session_id": "${CLAUDE_SESSION_ID:-fake-session}",
  "transcript_path": null,
  "status": "resolved",
  "verdict": "${verdict}",
  "resolution_reason": null,
  "current_iteration": 1,
  "max_rounds": 1,
  "debate": false,
  "roster_id": "fake-roster",
  "started_at": "2026-05-11T00:00:00Z",
  "ended_at": "2026-05-11T00:00:01Z"
}
JSON
  cat "${run_dir}/gate-state.json"
  exit "$exit_code"
}

case "$subcommand" in
  spawn-code-review|spawn-epic-verify)
    mkdir -p "$run_dir"
    case "$case_name" in
      pass_empty)
        write_output 1 "claude#1" "[]"
        ;;
      fail_two_reviewers)
        write_output 1 "claude#1" '[{"title":"Fix null handling","body":"Guard the optional value before use.","priority":1,"file_path":"src/example.py","line_start":10,"line_end":12,"confidence":0.93,"severity":"high"}]'
        write_output 1 "gemini#1" '[{"title":"Preserve reviewer attribution","body":"The formatter must show the originating reviewer.","priority":2,"file_path":"src/example.py","line_start":20,"line_end":20,"confidence":0.87,"severity":"medium"}]'
        ;;
      pass_no_resolve)
        write_output 1 "claude#1" "[]"
        ;;
      pass_empty_reviewers)
        mkdir -p "${run_dir}/iterations/1/round-1/reviewers"
        ;;
      debate_round_two)
        write_output 1 "claude#1" '[{"title":"Round one should be ignored","body":"Intermediate debate output.","priority":2,"file_path":"src/example.py","line_start":5,"line_end":5,"confidence":0.7,"severity":"medium"}]'
        write_output 2 "claude#1" '[{"title":"Round two final finding","body":"Only the final debate round should surface.","priority":1,"file_path":"src/example.py","line_start":30,"line_end":31,"confidence":0.95,"severity":"high"}]'
        ;;
      requires_decision)
        write_output 1 "claude#1" "[]"
        write_output 1 "gemini#1" "[]"
        ;;
      *)
        echo "unknown CERBERUS_FAKE_CASE=${case_name}" >&2
        exit 65
        ;;
    esac
    ;;
  wait|status)
    session_key=""
    session_id=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --session-key)
          session_key="${2:-}"
          shift 2
          ;;
        --session-id)
          session_id="${2:-}"
          shift 2
          ;;
        *)
          shift
          ;;
      esac
    done
    if [[ -n "$session_key" && "$session_key" != "$run_key" ]]; then
      echo "expected --session-key ${run_key}, got ${session_key}" >&2
      exit 66
    fi
    if [[ -z "$session_key" && -z "$session_id" ]]; then
      echo "expected --session-key ${run_key}" >&2
      exit 66
    fi

    case "$case_name" in
      pass_empty|pass_no_resolve|pass_empty_reviewers)
        write_gate_state "pass" 0
        ;;
      fail_two_reviewers|debate_round_two)
        write_gate_state "fail" 1
        ;;
      requires_decision)
        write_gate_state "requires_decision" 1
        ;;
      *)
        echo "unknown CERBERUS_FAKE_CASE=${case_name}" >&2
        exit 65
        ;;
    esac
    ;;
  resolve)
    exit 0
    ;;
  *)
    echo "unknown subcommand ${subcommand}" >&2
    exit 64
    ;;
esac
