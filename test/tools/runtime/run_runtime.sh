#!/usr/bin/env bash
# test/tools/runtime/run_runtime.sh
# Builds and runs the lib/Runtime unit test suite.
# Does NOT require a simulator or .bin file.
#
# Usage:
#   cd /path/to/Ascend-MLIR
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

# Build AscendCRuntime and runtime-session
echo "--- Building AscendCRuntime and runtime-session ---"
rm -f build/lib/libAscendCRuntime.a
cd build && cmake --build . --target AscendCRuntime runtime-session -j4 && cd ..

echo "--- Checking runtime-session CLI ---"
test -x build/bin/runtime-session
build/bin/runtime-session --help | grep -q "task graph runtime"

FAKE_ARTIFACT_ROOT="$(mktemp -d)"
trap 'rm -rf "$FAKE_ARTIFACT_ROOT"' EXIT
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
trap 'rm -rf "$FAKE_ARTIFACT_ROOT"; rm -f "$INVALID_STDERR" "$RUN_STDERR"' EXIT
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
    -o /tmp/test_runtime
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    test/tools/runtime/test_taskgraph_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o /tmp/test_taskgraph_runtime

# Run
echo "--- Running test_taskgraph_runtime ---"
/tmp/test_taskgraph_runtime
echo "--- Running test_runtime ---"
/tmp/test_runtime
