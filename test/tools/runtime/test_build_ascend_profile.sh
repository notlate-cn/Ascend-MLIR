#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env

HELP_OUTPUT="$(bash scripts/build.sh --help)"
grep -q -- "--build-ascend" <<<"${HELP_OUTPUT}"
grep -q -- "--enable-afir" <<<"${HELP_OUTPUT}"
grep -q -- "--disable-afir" <<<"${HELP_OUTPUT}"
grep -q -- "--disable-ccache" <<<"${HELP_OUTPUT}"

FAKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ascend-build-ccache.XXXXXX")"
mkdir -p "${FAKE_ROOT}/bin" "${FAKE_ROOT}/llvm/bin" "${FAKE_ROOT}/llvm/lib/cmake/mlir"
trap 'rm -rf "${TMP_BUILD:-}" "${FAKE_ROOT}"' EXIT

cat >"${FAKE_ROOT}/bin/ccache" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"${FAKE_ROOT}/bin/cmake" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${BUILD_DIR}/cmake.args"
touch "${BUILD_DIR}/build.ninja"
SH
cat >"${FAKE_ROOT}/bin/ninja" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"${FAKE_ROOT}/bin/python3" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"${FAKE_ROOT}/llvm/bin/llvm-config" <<'SH'
#!/usr/bin/env bash
case "$1" in
  --includedir) echo /tmp/fake-llvm/include ;;
  *) echo /tmp/fake-llvm ;;
esac
SH
chmod +x "${FAKE_ROOT}/bin/"* "${FAKE_ROOT}/llvm/bin/llvm-config"

(
  export PATH="${FAKE_ROOT}/bin:${PATH}"
  export BUILD_DIR="${FAKE_ROOT}/build-auto"
  export LLVM_BUILD_DIR="${FAKE_ROOT}/llvm"
  bash scripts/build.sh --build-ascend --jobs 1 >/dev/null
  grep -q -- "-DCMAKE_C_COMPILER_LAUNCHER=ccache" "${BUILD_DIR}/cmake.args"
  grep -q -- "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" "${BUILD_DIR}/cmake.args"
)

(
  export PATH="${FAKE_ROOT}/bin:${PATH}"
  export BUILD_DIR="${FAKE_ROOT}/build-disabled"
  export LLVM_BUILD_DIR="${FAKE_ROOT}/llvm"
  bash scripts/build.sh --build-ascend --disable-ccache --jobs 1 >/dev/null
  if grep -q -- "-DCMAKE_C_COMPILER_LAUNCHER=ccache" "${BUILD_DIR}/cmake.args"; then
    echo "--disable-ccache unexpectedly passed C compiler launcher" >&2
    exit 1
  fi
  if grep -q -- "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" "${BUILD_DIR}/cmake.args"; then
    echo "--disable-ccache unexpectedly passed CXX compiler launcher" >&2
    exit 1
  fi
)

TMP_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/ascend-build-profile.XXXXXX")"

cmake -G Ninja -S . -B "${TMP_BUILD}" \
  -DLLVM_BUILD_DIR="${LLVM_BUILD}" \
  -DASCEND_ENABLE_AFIR=OFF \
  -DASCEND_ENABLE_TESTS=ON \
  -DAFIR_ENABLE_BINDING_PYTHON=OFF >/dev/null

TARGETS="$(ninja -C "${TMP_BUILD}" -t targets all)"
grep -q "bin/ascend-mlir-opt" <<<"${TARGETS}"
grep -q "bin/ascend-mlir-translate" <<<"${TARGETS}"
grep -q "ascend-debug" <<<"${TARGETS}"
grep -q "lib/libAscendConversion.a" <<<"${TARGETS}"
grep -q "check-ascend-conversion" <<<"${TARGETS}"

if grep -q "bin/afir-opt" <<<"${TARGETS}"; then
  echo "AFIR-disabled profile unexpectedly exposes afir-opt" >&2
  exit 1
fi

if grep -q "bin/afir-translate" <<<"${TARGETS}"; then
  echo "AFIR-disabled profile unexpectedly exposes afir-translate" >&2
  exit 1
fi

if grep -q "AFIRPythonModules" <<<"${TARGETS}"; then
  echo "AFIR-disabled profile unexpectedly exposes AFIR Python modules" >&2
  exit 1
fi

if grep -q "check-afir-conversion" <<<"${TARGETS}"; then
  echo "AFIR-disabled profile unexpectedly exposes check-afir-conversion" >&2
  exit 1
fi
