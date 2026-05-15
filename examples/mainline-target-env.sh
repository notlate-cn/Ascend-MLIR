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
