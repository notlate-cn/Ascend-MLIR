#!/usr/bin/env bash
# run.sh – end-to-end pipeline: tm_tensor.attention → aclnn direct call → accuracy check
#
# Usage:
#   source examples/env_gser.sh   # or your env.sh that sets ASCEND_HOME_PATH / PATH
#   bash examples/aclnn-attn-e2e/run.sh
#
# Optional env vars:
#   BUILD_DIR   path to CMake build directory (default: <repo>/build)
#   WORK_DIR    scratch directory for generated files (default: /tmp/aclnn-attn-e2e)
#   ATOL        absolute tolerance for accuracy check (default: 0.1)
#   RTOL        relative tolerance for accuracy check (default: 0.05)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}}"
ATOL="${ATOL:-0.1}"
RTOL="${RTOL:-0.05}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/gser/Ascend/cann}"
CANN_ARCH="${CANN_ARCH:-x86_64-linux}"
CANN_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
CANN_DEVLIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/devlib"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
CANN_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"

mkdir -p "${WORK_DIR}"

echo "=== [1/6] Generate reference inputs and expected output ==="
python3 "${SCRIPT_DIR}/gen_ref.py" --out-dir "${WORK_DIR}"

echo ""
echo "=== [2/6] Lower tm_tensor.attention → aclnn.kind ==="
torch-opt \
    --convert-tm-tensor-attention \
    "${SCRIPT_DIR}/attention.mlir" \
    -o "${WORK_DIR}/network_aclnn_kind.mlir"
echo "  → ${WORK_DIR}/network_aclnn_kind.mlir"

echo ""
echo "=== [3/6] Finalize aclnn declarations (aclnn.op + aclnn.layout) ==="
afir-opt \
    --aclnn-finalize-decl \
    "${WORK_DIR}/network_aclnn_kind.mlir" \
    -o "${WORK_DIR}/network.mlir"
echo "  → ${WORK_DIR}/network.mlir"

echo ""
echo "=== [4/6] Generate network_host.cpp from network.mlir ==="
aclnn-backend \
    --input  "${WORK_DIR}/network.mlir" \
    --output "${WORK_DIR}/network_host.cpp"
echo "  → ${WORK_DIR}/network_host.cpp"

echo ""
echo "=== [5/6] Compile test harness ==="
g++ -std=c++17 -O2 \
    -I"${REPO_ROOT}/include" \
    -I"${ASCEND_HOME_PATH}/${CANN_ARCH}/include" \
    "${SCRIPT_DIR}/harness.cpp" \
    "${WORK_DIR}/network_host.cpp" \
    "${REPO_ROOT}/lib/Runtime/AclnnOps.cpp" \
    -L"${CANN_LIB}" \
    -L"${CANN_DEVLIB}" \
    -lascendcl \
    -lopapi_transformer \
    -lnnopbase \
    -lascend_hal \
    -Wl,-rpath,"${CANN_LIB}:${CANN_DEVLIB}" \
    -o "${WORK_DIR}/test_attention"
echo "  → ${WORK_DIR}/test_attention"

echo ""
echo "=== [6/6] Run accuracy test ==="
echo "  NOTE: requires a real NPU device (aclInit fails without hardware)."
echo "        Steps 1-5 verify the full MLIR→host-C++ pipeline."
export LD_LIBRARY_PATH="${CANN_LIB}:${CANN_DEVLIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"${WORK_DIR}/test_attention" \
    --q        "${WORK_DIR}/q.npy" \
    --k        "${WORK_DIR}/k.npy" \
    --v        "${WORK_DIR}/v.npy" \
    --mask     "${WORK_DIR}/mask.npy" \
    --init     "${WORK_DIR}/init.npy" \
    --expected "${WORK_DIR}/expected.npy" \
    --atol     "${ATOL}" \
    --rtol     "${RTOL}"