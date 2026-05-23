#!/usr/bin/env bash
# Shared target-aware schedule environment for mainline examples.

_ASCEND_MAINLINE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ASCEND_MAINLINE_REPO_ROOT="$(cd "${_ASCEND_MAINLINE_ENV_DIR}/.." && pwd)"

# shellcheck source=../scripts/resolve_ascend_env.sh
source "${_ASCEND_MAINLINE_REPO_ROOT}/scripts/resolve_ascend_env.sh"

if [[ -z "${CANN_ROOT:-}" ]]; then
  CANN_ROOT="$(resolve_ascend_home || true)"
fi

if [[ -z "${CANN_ROOT:-}" ]]; then
  echo "Set CANN_ROOT, ASCEND_HOME_PATH, or ASCEND_TOOLKIT_HOME for target-aware scheduling" >&2
  exit 2
fi

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_ROOT}}"

export PATH="${PATH}:${_ASCEND_MAINLINE_REPO_ROOT}/build/bin"

CANN_ARCH="${CANN_ARCH:-$(resolve_cann_arch_dir)}"
export CANN_ARCH
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
export SOC_VERSION

_ASCEND_MAINLINE_SIM_LIB="${CANN_ROOT}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
_ASCEND_MAINLINE_BASE_LIB="${CANN_ROOT}/${CANN_ARCH}/lib64"
_ASCEND_MAINLINE_DEVICE_LIB="${_ASCEND_MAINLINE_BASE_LIB}/device/lib64"
_ASCEND_MAINLINE_RUNTIME_STUB="${CANN_ROOT}/runtime/lib64/stub"
export LD_LIBRARY_PATH="${_ASCEND_MAINLINE_SIM_LIB}:${_ASCEND_MAINLINE_BASE_LIB}:${_ASCEND_MAINLINE_DEVICE_LIB}:${_ASCEND_MAINLINE_RUNTIME_STUB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
