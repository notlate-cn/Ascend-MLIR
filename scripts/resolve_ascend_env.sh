#!/usr/bin/env bash

resolve_ascend_home() {
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
