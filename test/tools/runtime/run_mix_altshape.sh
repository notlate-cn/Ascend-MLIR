#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
if [ -x /home/niu/code/llvm-project/llvm/build/bin/llvm-config ]; then
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
runtime_verify_setup_env
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

EXAMPLE_DIR="${PROJECT_ROOT}/examples/matmul-add-leakyrelu"

EXAMPLE_LOG="$(mktemp /tmp/runtime-mix-altshape.XXXXXX.log)"
cleanup() {
  rm -f "${EXAMPLE_LOG}"
}
trap cleanup EXIT

bash "${EXAMPLE_DIR}/run-mainline.sh" \
  --m 64 --k 128 --n 96 --seed 42 --log \
  >"${EXAMPLE_LOG}" 2>&1

grep -q '^session.backend=sim$' "${EXAMPLE_LOG}"
grep -q '^session.result=success$' "${EXAMPLE_LOG}"
grep -q '^session.validation=pass$' "${EXAMPLE_LOG}"
grep -q '^PASS$' "${EXAMPLE_LOG}"

echo "mix altshape pass"
