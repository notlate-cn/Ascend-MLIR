#!/usr/bin/env bash

try_source_default_ascend_env() {
  local candidate
  local candidates=(
    "${HOME}/Ascend/latest/set_env.sh"
    "${HOME}/Ascend/20260323_newest/ascend-toolkit/set_env.sh"
    "${HOME}/Ascend/20260323_newest-full/ascend-toolkit/set_env.sh"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck source=/dev/null
      source "${candidate}" >/dev/null 2>&1
      return 0
    fi
  done

  return 1
}

resolve_ascend_home() {
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    printf '%s\n' "${ASCEND_HOME_PATH}"
    return 0
  fi
  if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    printf '%s\n' "${ASCEND_TOOLKIT_HOME}"
    return 0
  fi
  try_source_default_ascend_env || true
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    printf '%s\n' "${ASCEND_HOME_PATH}"
    return 0
  fi
  if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    printf '%s\n' "${ASCEND_TOOLKIT_HOME}"
    return 0
  fi
  return 1
}

resolve_cann_arch_dir() {
  local machine
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64) printf '%s\n' "x86_64-linux" ;;
    aarch64|arm64) printf '%s\n' "aarch64-linux" ;;
    *)
      echo "Unsupported host architecture: ${machine}" >&2
      return 1
      ;;
  esac
}
