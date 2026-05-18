#!/usr/bin/env bash
# Verifies that the run-manifest-only runtime-session build does not link CANN
# compiler/simulator libraries into the process at startup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env

RUN_ONLY_BUILD_DIR="${RUNTIME_SESSION_RUN_ONLY_BUILD_DIR:-build-runtime-session-run-only}"
cmake -G Ninja -S . -B "${RUN_ONLY_BUILD_DIR}" \
  -DLLVM_BUILD_DIR="${LLVM_BUILD}" \
  -DASCEND_RUNTIME_SESSION_RUN_ONLY=ON >/dev/null
cmake --build "${RUN_ONLY_BUILD_DIR}" --target runtime-session -j2 >/dev/null

RUNTIME_SESSION="${RUN_ONLY_BUILD_DIR}/bin/runtime-session"
test -x "${RUNTIME_SESSION}"

NEEDED="$(readelf -d "${RUNTIME_SESSION}" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')"
for forbidden in \
  libtiling_api.so \
  libregister.so \
  libplatform.so \
  libunified_dlog.so \
  libruntime_camodel.so \
  libnpu_drv.so \
  libstars.so \
  libmodel_top.so \
  libascendcl.so \
  libge_common_base.so \
  libc_sec.so; do
  if printf '%s\n' "${NEEDED}" | grep -Fxq "${forbidden}"; then
    echo "Error: run-only runtime-session links forbidden startup dependency: ${forbidden}" >&2
    printf '%s\n' "${NEEDED}" >&2
    exit 1
  fi
done

HELP_OUTPUT="$("${RUNTIME_SESSION}" --help)"
if printf '%s\n' "${HELP_OUTPUT}" | grep -q -- "--kernel"; then
  echo "Error: run-only runtime-session help should not expose compile options" >&2
  exit 1
fi

STDERR_PATH="$(mktemp)"
trap 'rm -f "${STDERR_PATH}"' EXIT
if "${RUNTIME_SESSION}" 2>"${STDERR_PATH}"; then
  echo "Error: run-only runtime-session unexpectedly succeeded without --run-manifest" >&2
  exit 1
fi
grep -q "requires --run-manifest" "${STDERR_PATH}"
