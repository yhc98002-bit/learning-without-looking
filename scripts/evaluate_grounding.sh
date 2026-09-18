#!/usr/bin/env bash
# Evaluate one checkpoint on the grounding suite and its twin, sharding the
# manifest across the GPUs given to the script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${LWL_DATA:-${REPO_ROOT}/data}"

MODEL=""
OUT=""
INSTRUMENTS="suite twin"
GPUS="0"
CONDITION="real"
MAX_NEW_TOKENS=32
SEED=0

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_grounding.sh --model PATH --out DIR [options]

Options:
  --model PATH          Checkpoint or model directory to evaluate.
  --out DIR             Where predictions and metrics are written.
  --instruments "I ..." suite, twin, or both (default "suite twin").
  --gpus "N ..."        GPUs to shard over (default "0"); one shard per GPU.
  --condition NAME      Image condition (default real).
  --max-new-tokens N    Decoding budget (default 32).
  --seed N              Generation and noise seed (default 0).

Each instrument is read from grounding/<instrument>/manifest.jsonl in the data
directory (LWL_DATA, default data/). Decoding is greedy. Shards run in
parallel, one process per GPU, and write <out>/<instrument>/shards/shard_<i>.jsonl
and <out>/<instrument>/metrics/shard_<i>.json. A shard is skipped only when
both files are written and its predictions hold every manifest row assigned to
it; whatever an interrupted shard left behind is removed and the shard is run
again, so a run can be resumed with the same --gpus.
scripts/aggregate_evaluation.py combines the shards of one instrument into a
single metrics file.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --out) OUT="$2"; shift 2 ;;
      --instruments) INSTRUMENTS="$2"; shift 2 ;;
      --gpus) GPUS="$2"; shift 2 ;;
      --condition) CONDITION="$2"; shift 2 ;;
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
  if [[ -z "${GPUS//[[:space:]]/}" ]]; then
    echo "--gpus must name at least one GPU" >&2
    exit 2
  fi
  local instrument
  for instrument in ${INSTRUMENTS}; do
    case "${instrument}" in
      suite|twin) ;;
      *) echo "unknown instrument: ${instrument}" >&2; exit 2 ;;
    esac
  done
}

shard_metrics_path() {
  echo "$1/metrics/shard_$2.json"
}

count_rows() {
  grep -c '' "$1" || true
}

# Complete means the metrics exist, were written for the same number of shards,
# and the predictions hold the expected rows.
shard_complete() {
  local predictions="$1" metrics="$2" expected="$3" shards="$4"
  [[ -s "${metrics}" && -f "${predictions}" ]] || return 1
  [[ "$(count_rows "${predictions}")" -eq "${expected}" ]] || return 1
  grep -Eq "\"num_shards\": ${shards}(\.0)?,?$" "${metrics}"
}

# evaluate.py never overwrites, so an incomplete shard is cleared before a rerun.
clear_incomplete() {
  local path
  for path in "$@"; do
    if [[ -e "${path}" ]]; then
      echo "removing incomplete output: ${path}"
      rm -f "${path}"
    fi
  done
}

run_shard() {
  local manifest="$1" run_dir="$2" gpu="$3" shard="$4" shards="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python "${REPO_ROOT}/scripts/evaluate.py" \
    --model-path "${MODEL}" \
    --manifest "${manifest}" \
    --output "${run_dir}/shards/shard_${shard}.jsonl" \
    --metrics-output "$(shard_metrics_path "${run_dir}" "${shard}")" \
    --num-shards "${shards}" \
    --shard-index "${shard}" \
    --image-mode "${CONDITION}" \
    --image-cache-dir "${run_dir}/${CONDITION}_image_cache" \
    --seed "${SEED}" \
    --noise-seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    > "${run_dir}/logs/shard_${shard}.log" 2>&1
}

evaluate_instrument() {
  local instrument="$1"
  local instrument_dir="${DATA_ROOT}/grounding/${instrument}"
  local manifest="${instrument_dir}/manifest.jsonl"
  if [[ ! -f "${manifest}" ]]; then
    if [[ -f "${instrument_dir}.tar.gz" ]]; then
      echo "${instrument} is still packed; unpack it with: tar -xzf ${instrument_dir}.tar.gz -C ${DATA_ROOT}/grounding" >&2
    else
      echo "no such manifest: ${manifest}" >&2
    fi
    return 1
  fi
  local run_dir="${OUT}/${instrument}"
  mkdir -p "${run_dir}/shards" "${run_dir}/metrics" "${run_dir}/logs"

  local gpu_list=()
  read -r -a gpu_list <<< "${GPUS}"
  local shards="${#gpu_list[@]}"
  local total
  total="$(count_rows "${manifest}")"

  # Shards of a run over more GPUs would be aggregated along with this run's.
  local path index
  for path in "${run_dir}"/shards/shard_*.jsonl; do
    [[ -f "${path}" ]] || continue
    index="${path##*/shard_}"
    index="${index%.jsonl}"
    if (( index >= shards )); then
      echo "${run_dir} holds shards of a run over more GPUs; resume with the same --gpus or use a new --out" >&2
      return 1
    fi
  done

  local pids=()
  local shard predictions metrics
  for shard in "${!gpu_list[@]}"; do
    predictions="${run_dir}/shards/shard_${shard}.jsonl"
    metrics="$(shard_metrics_path "${run_dir}" "${shard}")"
    if shard_complete "${predictions}" "${metrics}" \
        "$(( (total - shard + shards - 1) / shards ))" "${shards}"; then
      echo "${instrument} shard ${shard}: complete, skipped"
      continue
    fi
    clear_incomplete "${predictions}" "${predictions}.partial" "${metrics}" "${metrics}.partial"
    run_shard "${manifest}" "${run_dir}" "${gpu_list[${shard}]}" "${shard}" "${shards}" &
    pids+=("$!")
    echo "${instrument} shard ${shard} on gpu ${gpu_list[${shard}]}: ${run_dir}/logs/shard_${shard}.log"
  done

  local status=0
  local pid
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${instrument}: at least one shard failed; see ${run_dir}/logs" >&2
  else
    echo "${instrument}: ${shards} shard(s) complete in ${run_dir}"
  fi
  return "${status}"
}

main() {
  parse_args "$@"
  local status=0
  local instrument
  for instrument in ${INSTRUMENTS}; do
    evaluate_instrument "${instrument}" || status=1
  done
  exit "${status}"
}

main "$@"
