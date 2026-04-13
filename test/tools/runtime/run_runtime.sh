#!/usr/bin/env bash
# test/tools/runtime/run_runtime.sh
# Builds the project and runs the task-graph runtime verification flow.
# Does NOT require a simulator or .bin file.
#
# Usage:
#   cd /path/to/Ascend-MLIR
#   export LLVM_BUILD_DIR=/path/to/llvm/build
#   bash test/tools/runtime/run_runtime.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running runtime tests"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"
source "${PROJECT_ROOT}/examples/env.sh" >/dev/null
CANN_ARCH="${CANN_ARCH:-$(resolve_cann_arch_dir)}"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib/davinci"
DEVICE_LIB="${ASCEND_HOME_PATH}/runtime/lib64/stub"

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "$LLVM_BUILD" ]; then
  exit 1
fi
LLVM_SOURCE_INCLUDE="$(cd "${LLVM_BUILD}/.." && pwd)/include"

if [ -f build/CMakeCache.txt ]; then
  CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt)"
  if [ -n "${CACHE_SOURCE_DIR}" ] && [ "${CACHE_SOURCE_DIR}" != "${PROJECT_ROOT}" ]; then
    echo "Recreating build/ because CMake cache points to ${CACHE_SOURCE_DIR}"
    rm -rf build
  fi
fi

cmake -S . -B build -DLLVM_BUILD_DIR="$LLVM_BUILD"

echo "--- Building focused runtime verification targets ---"
cd build && cmake --build . --target AscendCRuntime AFIRRuntimeCAPI runtime-session afir-opt afir-translate -j2 && cd ..

echo "--- Checking runtime-session CLI ---"
test -x build/bin/runtime-session
build/bin/runtime-session --help | grep -q "task graph runtime"

FAKE_ARTIFACT_ROOT="$(mktemp -d)"
INVALID_STDERR=""
RUN_STDERR=""
TEST_RUNTIME_BIN="$(mktemp /tmp/test_runtime.XXXXXX)"
TEST_TASKGRAPH_RUNTIME_BIN="$(mktemp /tmp/test_taskgraph_runtime.XXXXXX)"
TEST_CAPI_RUNTIME_BIN="$(mktemp /tmp/test_capi_runtime.XXXXXX)"
RUNTIME_SESSION_ARTIFACT_ROOT="$(mktemp -d)"
RUNTIME_SESSION_SECOND_ARTIFACT_ROOT="$(mktemp -d)"
RUNTIME_SESSION_RUN_MANIFEST="$(mktemp /tmp/runtime_session_run_manifest.XXXXXX.json)"
RUNTIME_SESSION_ACTUAL_OUTPUT="$(mktemp /tmp/runtime_session_actual.XXXXXX.npy)"
RUNTIME_SESSION_DAG_MANIFEST="$(mktemp /tmp/runtime_session_dag_manifest.XXXXXX.json)"
RUNTIME_SESSION_DAG_OUTPUT="$(mktemp /tmp/runtime_session_dag_actual.XXXXXX.npy)"
RUNTIME_SESSION_NPU_MANIFEST="$(mktemp /tmp/runtime_session_npu_manifest.XXXXXX.json)"
RUNTIME_SESSION_NPU_OUTPUT="$(mktemp /tmp/runtime_session_npu_actual.XXXXXX.npy)"
RUNTIME_SESSION_NPU_SUCCESS_STDOUT="$(mktemp /tmp/runtime_session_npu_success.XXXXXX.log)"
RUNTIME_SESSION_NPU_SUCCESS_STDERR="$(mktemp /tmp/runtime_session_npu_success_stderr.XXXXXX.log)"
NPU_STDERR=""
cleanup() {
  rm -rf "$FAKE_ARTIFACT_ROOT"
  rm -rf "$RUNTIME_SESSION_ARTIFACT_ROOT"
  rm -rf "$RUNTIME_SESSION_SECOND_ARTIFACT_ROOT"
  rm -f "$INVALID_STDERR" "$RUN_STDERR" "$TEST_RUNTIME_BIN" \
        "$TEST_TASKGRAPH_RUNTIME_BIN" "$TEST_CAPI_RUNTIME_BIN" \
        "$RUNTIME_SESSION_RUN_MANIFEST" \
        "$RUNTIME_SESSION_ACTUAL_OUTPUT" "$RUNTIME_SESSION_DAG_MANIFEST" \
        "$RUNTIME_SESSION_DAG_OUTPUT" "$RUNTIME_SESSION_NPU_MANIFEST" \
        "$RUNTIME_SESSION_NPU_OUTPUT" "$RUNTIME_SESSION_NPU_SUCCESS_STDOUT" \
        "$RUNTIME_SESSION_NPU_SUCCESS_STDERR" "$NPU_STDERR"
}
trap cleanup EXIT
mkdir -p "${FAKE_ARTIFACT_ROOT}/out"
cat > "${FAKE_ARTIFACT_ROOT}/out/manifest.txt" <<'EOF'
kernel_name=fake_kernel
soc_version=Ascend910B1
kernel_kind=vec
device_binary_path=fake.bin
EOF

echo "--- Checking runtime-session planning path ---"
PLAN_OUTPUT="$(build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}")"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\[0\]=main"

echo "--- Checking runtime-session negative paths ---"
INVALID_STDERR="$(mktemp)"
RUN_STDERR="$(mktemp)"
if build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}/missing" 2>"${INVALID_STDERR}"; then
  echo "Error: invalid artifact root unexpectedly succeeded" >&2
  exit 1
fi
grep -q "cannot access artifact root" "${INVALID_STDERR}"

if build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}" --run 2>"${RUN_STDERR}"; then
  echo "Error: runtime-session --run unexpectedly succeeded" >&2
  exit 1
fi
grep -q "simulation path requires at least one output binding" "${RUN_STDERR}"

echo "--- Checking runtime-session positive vec simulation path ---"
bash examples/relu-broadcast-transpose/run.sh >/tmp/runtime_session_example.log 2>&1
build/bin/runtime-session \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --name relu_transpose_broadcast_add \
  --output "${RUNTIME_SESSION_ARTIFACT_ROOT}" \
  >/tmp/runtime_session_compile.log 2>&1
test -f "${RUNTIME_SESSION_ARTIFACT_ROOT}/out/manifest.txt"
build/bin/runtime-session \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --name relu_transpose_broadcast_add \
  --output "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}" \
  >/tmp/runtime_session_compile_consumer.log 2>&1
test -f "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}/out/manifest.txt"

cat > "${RUNTIME_SESSION_RUN_MANIFEST}" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${RUNTIME_SESSION_ARTIFACT_ROOT}",
  "inputs": [
    { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data0.npy" },
    { "name": "data1", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data1.npy" }
  ],
  "outputs": [
    {
      "name": "out",
      "path": "${RUNTIME_SESSION_ACTUAL_OUTPUT}",
      "shape": [500, 640],
      "dtype": "f16"
    }
  ],
  "tiling": {
    "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/tiling_space.json",
    "params": "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640"
  },
  "block_dim": 8,
  "workspace_size": 16777216,
  "profiling": true
}
EOF

build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_RUN_MANIFEST}" \
  --run >/tmp/runtime_session_run.log 2>&1
test -f "${RUNTIME_SESSION_ACTUAL_OUTPUT}"
grep -q '^session.profile.session_id=' /tmp/runtime_session_run.log
grep -q '^session.profile.count=' /tmp/runtime_session_run.log

echo "--- Checking runtime-session DAG simulation path ---"
cat > "${RUNTIME_SESSION_DAG_MANIFEST}" <<EOF
{
  "backend": "sim",
  "artifact_root": "${RUNTIME_SESSION_ARTIFACT_ROOT}",
  "tasks": [
    {
      "task_id": "producer",
      "inputs": [
        { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data0.npy" },
        { "name": "data1", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data1.npy" }
      ],
      "outputs": [
        { "name": "mid", "shape": [500, 640], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/tiling_space.json",
        "params": "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640"
      },
      "block_dim": 8,
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "consumer",
      "dependencies": ["producer"],
      "artifact_root": "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}",
      "inputs": [
        { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data0.npy" },
        { "name": "data1", "source": "task_output", "upstream_task": "producer", "upstream_output": "mid" }
      ],
      "outputs": [
        { "name": "out", "path": "${RUNTIME_SESSION_DAG_OUTPUT}", "shape": [500, 640], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/tiling_space.json",
        "params": "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640"
      },
      "block_dim": 8,
      "workspace_size": 16777216,
      "profiling": true
    }
  ]
}
EOF

PLAN_OUTPUT="$(build/bin/runtime-session --run-manifest "${RUNTIME_SESSION_DAG_MANIFEST}")"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[0\\]=producer"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[1\\]=consumer"
build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_DAG_MANIFEST}" \
  --run >/tmp/runtime_session_dag_run.log 2>&1
test -f "${RUNTIME_SESSION_DAG_OUTPUT}"
grep -q '^session.profile.session_id=' /tmp/runtime_session_dag_run.log
grep -q '^session.profile.count=' /tmp/runtime_session_dag_run.log

echo "--- Checking runtime-session NPU path reaches unified backend ---"
cat > "${RUNTIME_SESSION_NPU_MANIFEST}" <<EOF
{
  "backend": "npu",
  "artifact_root": "${FAKE_ARTIFACT_ROOT}",
  "tasks": [
    {
      "task_id": "main",
      "outputs": [
        { "name": "out", "path": "${RUNTIME_SESSION_NPU_OUTPUT}", "shape": [4], "dtype": "f16" }
      ]
    }
  ]
}
EOF
NPU_STDERR="$(mktemp)"
if build/bin/runtime-session --run-manifest "${RUNTIME_SESSION_NPU_MANIFEST}" --run 2>"${NPU_STDERR}"; then
  echo "Error: runtime-session npu path unexpectedly succeeded" >&2
  exit 1
fi
grep -q '^session.backend=npu' "${NPU_STDERR}"
grep -q '^session.result=error' "${NPU_STDERR}"
grep -q '^session.error_stage=executor_initialize' "${NPU_STDERR}"

echo "--- Checking runtime-session NPU mock success path ---"
build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_NPU_MANIFEST}" \
  --testing-driver npu-success \
  --run >"${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}" 2>"${RUNTIME_SESSION_NPU_SUCCESS_STDERR}"
grep -q '^session.backend=npu' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.result=success' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile.session_id=' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile.count=1' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile\[0\]=' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"

# Compile test drivers
echo "--- Compiling runtime tests ---"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_RUNTIME_BIN"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_taskgraph_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_TASKGRAPH_RUNTIME_BIN"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_capi_runtime.cpp \
    build/lib/libAFIRRuntimeCAPI.so \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_CAPI_RUNTIME_BIN"

# Run
echo "--- Running test_taskgraph_runtime ---"
"$TEST_TASKGRAPH_RUNTIME_BIN"
echo "--- Running test_capi_runtime ---"
LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "$TEST_CAPI_RUNTIME_BIN"
echo "--- Running test_runtime ---"
if "$TEST_RUNTIME_BIN"; then
  echo "--- Running SimBackend smoke baseline ---"
  bash test/tools/runtime/run_simbackend_smoke.sh
  exit 0
else
  STATUS=$?
  if [ "$STATUS" -eq 139 ]; then
    echo "Note: /tmp/test_runtime still segfaults in xvm after the task-graph runtime checks pass." >&2
  fi
  exit "$STATUS"
fi
