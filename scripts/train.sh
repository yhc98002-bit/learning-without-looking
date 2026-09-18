#!/usr/bin/env bash
# Train one run with the patched trainer from artifacts/repos/EasyR1.
# Runs trained in segments have one config per segment; run them in order.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYR1="${LWL_EASYR1:-${REPO_ROOT}/artifacts/repos/EasyR1}"

usage() {
  cat <<'EOF'
Usage: scripts/train.sh CONFIG_NAME [key=value ...]

Runs configs/train/CONFIG_NAME.yaml, for example constructed-standard-7b-run1.
Extra key=value arguments are passed to the trainer and override config fields,
for example data.rollout_batch_size=240.

Environment:
  CUDA_VISIBLE_DEVICES   GPUs for this run; the runs used all eight of one node.
  LWL_DATA               Must be unset or resolve to data/ in the repository:
                         the configs and training files name data/... relative
                         to it. Link a data directory kept elsewhere there.
  LWL_EASYR1             Trainer tree. Defaults to artifacts/repos/EasyR1.
  LWL_REWARD_SHADOW_LOG  Where the reward records its parser disagreements. The
                         reward requires one; it defaults to
                         logs/CONFIG_NAME/reward_shadow.jsonl.
  RAY_TMPDIR             Ray runtime directory; defaults to a per-run directory
                         under TMPDIR.
EOF
}

main() {
  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
  fi
  case "$1" in
    -h|--help) usage; exit 0 ;;
  esac

  local name="$1"
  shift
  local config="${REPO_ROOT}/configs/train/${name}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "no such config: ${config}" >&2
    exit 2
  fi
  if [[ ! -d "${EASYR1}" ]]; then
    echo "trainer tree is absent: ${EASYR1} (run scripts/setup_easyr1.sh)" >&2
    exit 2
  fi

  # The configs and the rows of the training files name data as data/..., relative
  # to the repository, so a data directory elsewhere has to be linked there.
  if [[ -n "${LWL_DATA:-}" && "$(realpath -m "${LWL_DATA}")" != "$(realpath -m "${REPO_ROOT}/data")" ]]; then
    echo "the training files read data/ inside the repository, not LWL_DATA=${LWL_DATA};" >&2
    echo "link it first: ln -s \"$(realpath -m "${LWL_DATA}")\" \"${REPO_ROOT}/data\"" >&2
    exit 2
  fi

  cd "${REPO_ROOT}"
  # The patched framework reads the attention implementation from the
  # environment; the runs used sdpa rather than flash-attention.
  export EASYR1_ATTN_IMPLEMENTATION=sdpa
  export LWL_REWARD_SHADOW_LOG="${LWL_REWARD_SHADOW_LOG:-${REPO_ROOT}/logs/${name}/reward_shadow.jsonl}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/lwl-ray-${name}}"
  export RAY_DEDUP_LOGS=0
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${EASYR1}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  mkdir -p "${RAY_TMPDIR}"

  echo "config: ${config}"
  echo "trainer: ${EASYR1}"
  echo "reward shadow log: ${LWL_REWARD_SHADOW_LOG}"
  exec python -u -m verl.trainer.main "config=${config}" "$@"
}

main "$@"
