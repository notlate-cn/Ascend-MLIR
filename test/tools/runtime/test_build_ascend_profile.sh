#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env

HELP_OUTPUT="$(bash scripts/build.sh --help)"
grep -q -- "--build-ascend" <<<"${HELP_OUTPUT}"
grep -q -- "--enable-afir" <<<"${HELP_OUTPUT}"
grep -q -- "--disable-afir" <<<"${HELP_OUTPUT}"

TMP_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/ascend-build-profile.XXXXXX")"
trap 'rm -rf "${TMP_BUILD}"' EXIT

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
