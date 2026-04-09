#!/usr/bin/env bash

resolve_llvm_build_dir() {
  if [[ -n "${LLVM_BUILD_DIR:-}" ]]; then
    printf '%s\n' "${LLVM_BUILD_DIR}"
    return 0
  fi

  local script_dir project_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_root="$(cd "${script_dir}/.." && pwd)"

  local candidates=(
    "${project_root}/externals/llvm-project/build"
    "${project_root}/../llvm-project/llvm/build"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}/bin/llvm-config" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

require_llvm_build_dir() {
  local llvm_build
  llvm_build="$(resolve_llvm_build_dir || true)"
  if [[ -z "${llvm_build}" ]]; then
    echo "LLVM build dir is not configured. Set LLVM_BUILD_DIR or provide bin/llvm-config under a supported default location." >&2
    return 1
  fi
  if [[ ! -x "${llvm_build}/bin/llvm-config" ]]; then
    echo "LLVM build dir is invalid (missing bin/llvm-config): ${llvm_build}" >&2
    return 1
  fi
  printf '%s\n' "${llvm_build}"
}
