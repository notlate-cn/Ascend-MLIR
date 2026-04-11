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

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "$LLVM_BUILD" ]; then
  exit 1
fi

if [ -f build/CMakeCache.txt ]; then
  CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt)"
  if [ -n "${CACHE_SOURCE_DIR}" ] && [ "${CACHE_SOURCE_DIR}" != "${PROJECT_ROOT}" ]; then
    echo "Recreating build/ because CMake cache points to ${CACHE_SOURCE_DIR}"
    rm -rf build
  fi
fi

cmake -S . -B build -DLLVM_BUILD_DIR="$LLVM_BUILD"

echo "--- Building focused runtime verification targets ---"
cd build && cmake --build . --target AscendCRuntime runtime-session -j2 && cd ..

echo "--- Checking runtime-session CLI ---"
test -x build/bin/runtime-session
build/bin/runtime-session --help | grep -q "task graph runtime"

FAKE_ARTIFACT_ROOT="$(mktemp -d)"
INVALID_STDERR=""
RUN_STDERR=""
TEST_RUNTIME_BIN="$(mktemp /tmp/test_runtime.XXXXXX)"
TEST_TASKGRAPH_RUNTIME_BIN="$(mktemp /tmp/test_taskgraph_runtime.XXXXXX)"
cleanup() {
  rm -rf "$FAKE_ARTIFACT_ROOT"
  rm -f "$INVALID_STDERR" "$RUN_STDERR" "$TEST_RUNTIME_BIN" "$TEST_TASKGRAPH_RUNTIME_BIN"
}
trap cleanup EXIT
mkdir -p "${FAKE_ARTIFACT_ROOT}/out"
cat > "${FAKE_ARTIFACT_ROOT}/out/manifest.txt" <<'EOF'
kernel_name=fake_kernel
soc_version=Ascend910B1
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
grep -q "task I/O binding is not implemented" "${RUN_STDERR}"

# Compile test drivers
echo "--- Compiling runtime tests ---"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    test/tools/runtime/test_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o "$TEST_RUNTIME_BIN"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    test/tools/runtime/test_taskgraph_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o "$TEST_TASKGRAPH_RUNTIME_BIN"

# Run
echo "--- Running test_taskgraph_runtime ---"
"$TEST_TASKGRAPH_RUNTIME_BIN"
echo "--- Running test_runtime ---"
if "$TEST_RUNTIME_BIN"; then
  exit 0
else
  STATUS=$?
  if [ "$STATUS" -eq 139 ]; then
    echo "Note: /tmp/test_runtime still segfaults in xvm after the task-graph runtime checks pass." >&2
  fi
  exit "$STATUS"
fi
