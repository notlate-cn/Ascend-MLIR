#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/versions.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/versions.env"
fi

TAG="${ASCEND_MLIR_CI_IMAGE:-${ASCEND_MLIR_CI_DEFAULT_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04}}"
BASE_IMAGE="${ASCEND_MLIR_CI_BASE_IMAGE:-ubuntu:22.04}"
LLVM_BUILD_SRC="${ASCEND_MLIR_CI_EMBED_LLVM_BUILD_DIR:-}"
PLATFORM="${ASCEND_MLIR_CI_DOCKER_PLATFORM:-linux/arm64}"
BUILD_LLVM=0
LLVM_REPO_URL="${ASCEND_MLIR_CI_LLVM_REPO_URL:-https://github.com/llvm/llvm-project.git}"
LLVM_REF="${ASCEND_MLIR_CI_LLVM_REF:-2078da43e25a4623cab2d0d60decddf709aaea28}"
LLVM_BUILD_TYPE="${ASCEND_MLIR_CI_LLVM_BUILD_TYPE:-Release}"
LLVM_ENABLE_ASSERTIONS="${ASCEND_MLIR_CI_LLVM_ENABLE_ASSERTIONS:-ON}"
LLVM_JOBS="${ASCEND_MLIR_CI_LLVM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
SAVE_TAR=""
PUSH=0
PROVENANCE="${ASCEND_MLIR_CI_DOCKER_PROVENANCE:-false}"
SBOM="${ASCEND_MLIR_CI_DOCKER_SBOM:-false}"

usage() {
  cat <<'EOF'
Usage: build-aarch64-image.sh [OPTIONS]

Prebuild the generic aarch64 real-NPU validation image. Run this from an
arm64-capable Docker environment such as xvm/OrbStack, native arm64 Linux, or
an arm64 CI builder.

Options:
  --tag IMAGE                    Image tag. Default: ascend-mlir-builder:aarch64-ubuntu22.04
  --base-image IMAGE             Base image. Default: ubuntu:22.04
  --with-llvm                    Clone and build pinned LLVM/MLIR into /opt/llvm/build.
  --llvm-repo URL                LLVM repository URL.
  --llvm-ref REF                 LLVM ref, tag, branch, or commit.
  --llvm-build-type TYPE         LLVM CMAKE_BUILD_TYPE. Default: Release.
  --llvm-enable-assertions BOOL  LLVM_ENABLE_ASSERTIONS. Default: ON.
  --llvm-jobs N                  Parallel jobs for LLVM build.
  --embed-llvm-build-dir DIR     Copy an existing arm64 LLVM build into /opt/llvm/build.
  --platform PLATFORM            docker buildx --platform value. Default: linux/arm64.
  --save-tar FILE                Save the built image to a tar file.
  --push                         Push the built image after build.
  --provenance BOOL              docker buildx provenance setting. Default: false.
  --sbom BOOL                    docker buildx SBOM setting. Default: false.
  --help                         Show this help.

The image intentionally does not include the Ascend host driver. Mount driver,
device nodes, and the CANN toolkit from the real-NPU host at job runtime.

Without --with-llvm or --embed-llvm-build-dir, jobs must mount LLVM via
--llvm-build-dir or set ASCEND_MLIR_CI_BUILD_LLVM=1. For normal shared
validation, prebuild once with LLVM embedded so each job only builds the
checked-out Ascend-MLIR source.

The default build disables BuildKit provenance/SBOM attestations because some
registries, including Huawei Cloud SWR, reject the attestation manifest form.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="$2"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --with-llvm)
      BUILD_LLVM=1
      if [[ "${TAG}" == "${ASCEND_MLIR_CI_DEFAULT_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04}" ]]; then
        TAG="${ASCEND_MLIR_CI_DEFAULT_LLVM_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04-llvm21}"
      fi
      shift
      ;;
    --llvm-repo)
      LLVM_REPO_URL="$2"
      shift 2
      ;;
    --llvm-ref)
      LLVM_REF="$2"
      shift 2
      ;;
    --llvm-build-type)
      LLVM_BUILD_TYPE="$2"
      shift 2
      ;;
    --llvm-enable-assertions)
      LLVM_ENABLE_ASSERTIONS="$2"
      shift 2
      ;;
    --llvm-jobs)
      LLVM_JOBS="$2"
      shift 2
      ;;
    --embed-llvm-build-dir)
      LLVM_BUILD_SRC="$2"
      shift 2
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --save-tar)
      SAVE_TAR="$2"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --provenance)
      PROVENANCE="$2"
      shift 2
      ;;
    --sbom)
      SBOM="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the image" >&2
  exit 1
fi
if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required to build the image" >&2
  exit 1
fi

if [[ "${BUILD_LLVM}" == "1" && -n "${LLVM_BUILD_SRC}" ]]; then
  echo "--with-llvm and --embed-llvm-build-dir are mutually exclusive" >&2
  exit 2
fi

BUILD_CONTEXT="${SCRIPT_DIR}"
TMP_CONTEXT=""
cleanup() {
  if [[ -n "${TMP_CONTEXT}" ]]; then
    rm -rf "${TMP_CONTEXT}"
  fi
}
trap cleanup EXIT

if [[ -n "${LLVM_BUILD_SRC}" ]]; then
  if [[ ! -d "${LLVM_BUILD_SRC}/lib/cmake/mlir" ]]; then
    echo "LLVM build dir is invalid: ${LLVM_BUILD_SRC}" >&2
    echo "expected ${LLVM_BUILD_SRC}/lib/cmake/mlir" >&2
    exit 1
  fi

  TMP_CONTEXT="$(mktemp -d)"
  rsync -a --exclude llvm-build "${SCRIPT_DIR}/" "${TMP_CONTEXT}/"
  mkdir -p "${TMP_CONTEXT}/llvm-build"
  rsync -a "${LLVM_BUILD_SRC}/" "${TMP_CONTEXT}/llvm-build/"
  BUILD_CONTEXT="${TMP_CONTEXT}"
fi

DOCKER_ARGS=()
if [[ -n "${PLATFORM}" ]]; then
  DOCKER_ARGS+=(--platform "${PLATFORM}")
fi

docker buildx build \
  "${DOCKER_ARGS[@]}" \
  --load \
  --provenance="${PROVENANCE}" \
  --sbom="${SBOM}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg BUILD_LLVM="${BUILD_LLVM}" \
  --build-arg LLVM_REPO_URL="${LLVM_REPO_URL}" \
  --build-arg LLVM_REF="${LLVM_REF}" \
  --build-arg LLVM_BUILD_TYPE="${LLVM_BUILD_TYPE}" \
  --build-arg LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS}" \
  --build-arg LLVM_JOBS="${LLVM_JOBS}" \
  -f "${SCRIPT_DIR}/Dockerfile.builder" \
  -t "${TAG}" \
  "${BUILD_CONTEXT}"

if [[ "${PUSH}" == "1" ]]; then
  docker push "${TAG}"
fi

if [[ -n "${SAVE_TAR}" ]]; then
  mkdir -p "$(dirname "${SAVE_TAR}")"
  docker save "${TAG}" -o "${SAVE_TAR}"
fi
