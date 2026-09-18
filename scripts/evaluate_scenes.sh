#!/usr/bin/env bash
# Evaluate one checkpoint on the constructed scenes: every cue level of a scene
# split, under each image condition, with the decoding the paper reports.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${LWL_DATA:-${REPO_ROOT}/data}"

MODEL=""
OUT=""
SPLIT="development"
LEVELS="l3 l2 l1 probe"
CONDITIONS="real gray"
MAX_NEW_TOKENS=32
SEED=0

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_scenes.sh --model PATH --out DIR [options]

Options:
  --model PATH         Checkpoint or model directory to evaluate.
  --out DIR            Where predictions and metrics are written.
  --split NAME         training, development or confirmatory (default development).
  --levels "L ..."     Cue levels, as they are named in the data files
                       (default "l3 l2 l1 probe"):
                         l1     cued readout
                         l2     find and bind
                         l3     discovery
                         probe  identification probe
  --conditions "C ..." Image conditions (default "real gray").
  --max-new-tokens N   Decoding budget (default 32).
  --seed N             Generation and noise seed (default 0).

Decoding is greedy. One cell is one manifest under one condition. Scenes are
read from the data directory (LWL_DATA, default data/) under
scenes/<split>/manifest_*_<level>.jsonl. A cell is skipped only when both its
predictions and its metrics are written and the predictions hold one row per
manifest row; whatever an interrupted cell left behind is removed and the cell
is run again, so a sweep can be resumed. The GPU is chosen with
CUDA_VISIBLE_DEVICES.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --out) OUT="$2"; shift 2 ;;
      --split) SPLIT="$2"; shift 2 ;;
      --levels) LEVELS="$2"; shift 2 ;;
      --conditions) CONDITIONS="$2"; shift 2 ;;
      --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
      --seed) SEED="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
  done
  if [[ -z "${MODEL}" || -z "${OUT}" ]]; then
    usage >&2
    exit 2
  fi
}

count_rows() {
  grep -c '' "$1" || true
}

# Complete means the metrics exist and the predictions hold the expected rows.
output_complete() {
  local predictions="$1" metrics="$2" expected="$3"
  [[ -s "${metrics}" && -f "${predictions}" ]] || return 1
  [[ "$(count_rows "${predictions}")" -eq "${expected}" ]]
}

# evaluate.py never overwrites, so an incomplete cell is cleared before a rerun.
clear_incomplete() {
  local path
  for path in "$@"; do
    if [[ -e "${path}" ]]; then
      echo "removing incomplete output: ${path}"
      rm -f "${path}"
    fi
  done
}

evaluate_cell() {
  local manifest="$1" condition="$2"
  local cell
  cell="$(basename "${manifest}" .jsonl)"
  cell="${cell#manifest_}"
  local out_dir="${OUT}/${condition}/${cell}"
  local predictions="${out_dir}/predictions.jsonl"
  local metrics="${out_dir}/metrics.json"
  if output_complete "${predictions}" "${metrics}" "$(count_rows "${manifest}")"; then
    echo "skip ${condition}/${cell}: complete"
    return 0
  fi
  clear_incomplete "${predictions}" "${predictions}.partial" "${metrics}" "${metrics}.partial"
  echo "run ${condition}/${cell}"
  python "${REPO_ROOT}/scripts/evaluate.py" \
    --model-path "${MODEL}" \
    --manifest "${manifest}" \
    --output "${predictions}" \
    --metrics-output "${metrics}" \
    --image-mode "${condition}" \
    --image-cache-dir "${OUT}/${condition}/image_cache" \
    --seed "${SEED}" \
    --noise-seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
}

main() {
  parse_args "$@"
  local scene_dir="${DATA_ROOT}/scenes/${SPLIT}"
  if [[ ! -d "${scene_dir}" ]]; then
    if [[ -f "${scene_dir}.tar.gz" ]]; then
      echo "scene split is still packed; unpack it with: tar -xzf ${scene_dir}.tar.gz -C ${DATA_ROOT}/scenes" >&2
    else
      echo "no such scene split: ${scene_dir}" >&2
    fi
    exit 2
  fi

  local status=0
  local condition level manifest found
  for condition in ${CONDITIONS}; do
    for level in ${LEVELS}; do
      found=0
      for manifest in "${scene_dir}"/manifest_*_"${level}".jsonl; do
        [[ -f "${manifest}" ]] || continue
        found=1
        evaluate_cell "${manifest}" "${condition}" || status=1
      done
      if [[ "${found}" -eq 0 ]]; then
        echo "no manifests for level ${level} in ${scene_dir}" >&2
        status=1
      fi
    done
  done
  exit "${status}"
}

main "$@"
