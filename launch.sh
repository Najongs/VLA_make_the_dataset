#!/usr/bin/env bash
set -euo pipefail

START=${1:-1}            # 시작 숫자 (기본 1)
COUNT=${2:-20}           # 반복 횟수 (기본 20)
SLEEP_SECONDS=${SLEEP_SECONDS:-5}  # 반복 사이 딜레이(초)
PY=${PYTHON:-python3}
SCRIPT=${SCRIPT:-ZED_Capture.py}

ordinal() {
  local n="$1"
  local m=$((n % 100))
  local s=$((n % 10))
  if ((m >= 11 && m <= 13)); then
    echo "${n}th"
  else
    case $s in
      1) echo "${n}st" ;;
      2) echo "${n}nd" ;;
      3) echo "${n}rd" ;;
      *) echo "${n}th" ;;
    esac
  fi
}

END=$((START + COUNT - 1))
for ((i=START; i<=END; i++)); do
  tag="$(ordinal "$i")"
  echo "[INFO] $(date '+%F %T') Run $i/$END -> tag=${tag}"
  "$PY" "$SCRIPT" --run-tag "$tag"
  if (( i < END )); then
    echo "[INFO] $(date '+%F %T') Sleeping ${SLEEP_SECONDS}s before next run..."
    sleep "${SLEEP_SECONDS}"
  fi
done
