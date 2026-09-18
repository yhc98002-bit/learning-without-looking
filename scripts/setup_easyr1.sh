#!/usr/bin/env bash
# Clone the training framework into artifacts/repos/EasyR1, pin it to the
# revision the runs used, and apply the patches in patches/easyr1.
# Re-running the script is safe: each step is skipped when it is already done.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/artifacts/repos/EasyR1"
PATCH_DIR="${REPO_ROOT}/patches/easyr1"
UPSTREAM="https://github.com/hiyouga/EasyR1"
REVISION="dd71bbd252694f5f850213eec15795b6b88d9fea"

# Order matters for the first four: the caption patch edits code the image-condition
# patch adds, and the grid and hash patches edit code the caption patch adds.
# Each entry is patch|file|marker, where the marker is a string the patch adds:
# a patch whose marker is already in the tree has been applied.
PATCHES=(
  "easyr1_image_condition_patch.diff|verl/utils/dataset.py|IMAGE_CONDITIONS = "
  "easyr1_caption_condition_patch.diff|verl/utils/dataset.py|caption_store_paths"
  "easyr1_multimodal_grid_patch.diff|verl/utils/dataset.py|rollout_images = []"
  "easyr1_caption_pil_hash_patch.diff|verl/utils/dataset.py|caption_by_pixel_hash"
  "easyr1_resume_safe_logger_patch.diff|verl/utils/logger/logger.py|Preserving existing EasyR1 file logger artifact during resume"
  "easyr1_sdpa_patch.diff|verl/workers/fsdp_workers.py|EASYR1_ATTN_IMPLEMENTATION"
)

PATCHED_FILES=(
  verl/trainer/config.py
  verl/trainer/data_loader.py
  verl/utils/dataset.py
  verl/utils/logger/logger.py
  verl/workers/fsdp_workers.py
)

usage() {
  cat <<EOF
Usage: scripts/setup_easyr1.sh [-h]

Prepares the patched trainer used by scripts/train.sh:
  1. clones ${UPSTREAM} into artifacts/repos/EasyR1
  2. checks out ${REVISION}
  3. applies the patches in patches/easyr1
EOF
}

clone_upstream() {
  if [[ -d "${TARGET}/.git" ]]; then
    echo "checkout present: ${TARGET}"
    return
  fi
  mkdir -p "$(dirname "${TARGET}")"
  git clone "${UPSTREAM}" "${TARGET}"
  echo "cloned ${UPSTREAM} into ${TARGET}"
}

pin_revision() {
  if [[ "$(git -C "${TARGET}" rev-parse HEAD)" == "${REVISION}" ]]; then
    echo "revision already pinned: ${REVISION}"
    return
  fi
  if ! git -C "${TARGET}" diff --quiet; then
    echo "checkout is modified and is not at ${REVISION}: ${TARGET}" >&2
    exit 1
  fi
  git -C "${TARGET}" checkout --detach --quiet "${REVISION}"
  echo "checked out ${REVISION}"
}

apply_patch() {
  local name="$1" marker_file="$2" marker="$3"
  local patch="${PATCH_DIR}/${name}"
  if [[ ! -f "${patch}" ]]; then
    echo "missing patch: ${patch}" >&2
    exit 1
  fi
  if grep -Fq "${marker}" "${TARGET}/${marker_file}"; then
    echo "already applied: ${name}"
    return
  fi
  git -C "${TARGET}" apply --check "${patch}"
  git -C "${TARGET}" apply "${patch}"
  grep -Fq "${marker}" "${TARGET}/${marker_file}"
  echo "applied: ${name}"
}

compile_patched_files() {
  local paths=()
  local name
  for name in "${PATCHED_FILES[@]}"; do
    paths+=("${TARGET}/${name}")
  done
  python3 -m py_compile "${paths[@]}"
  echo "compiled ${#PATCHED_FILES[@]} patched files"
}

main() {
  if [[ $# -gt 0 ]]; then
    case "$1" in
      -h|--help) usage; exit 0 ;;
      *) usage >&2; exit 2 ;;
    esac
  fi
  clone_upstream
  pin_revision
  local entry name marker_file marker
  for entry in "${PATCHES[@]}"; do
    IFS='|' read -r name marker_file marker <<< "${entry}"
    apply_patch "${name}" "${marker_file}" "${marker}"
  done
  compile_patched_files
  echo "trainer ready: ${TARGET}"
}

main "$@"
