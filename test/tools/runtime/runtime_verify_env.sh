#!/usr/bin/env bash

runtime_verify_setup_env() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${script_dir}/../../.." && pwd)"
  export PROJECT_ROOT
  cd "${PROJECT_ROOT}"

  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

  ASCEND_HOME="$(resolve_ascend_home || true)"
  if [ -z "${ASCEND_HOME}" ]; then
    echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running runtime tests" >&2
    return 1
  fi
  export ASCEND_HOME_PATH="${ASCEND_HOME}"

  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/examples/env.sh" >/dev/null

  CANN_ARCH="${CANN_ARCH:-$(resolve_cann_arch_dir)}"
  export CANN_ARCH
  SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
  export SOC_VERSION
  ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
  export ASCEND_LIB64
  SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
  export SOC_SIM_LIB
  DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"
  export DAV_SIM_VERSION
  DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${DAV_SIM_VERSION}/lib"
  export DAV_SIM_LIB
  DEVICE_STUB_LIB="${ASCEND_HOME_PATH}/runtime/lib64/stub"
  export DEVICE_STUB_LIB
  DEVICE_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64"
  export DEVICE_LIB64

  LLVM_BUILD="$(require_llvm_build_dir || true)"
  if [ -z "${LLVM_BUILD}" ]; then
    return 1
  fi
  export LLVM_BUILD
  LLVM_SOURCE_INCLUDE="$(cd "${LLVM_BUILD}/.." && pwd)/include"
  export LLVM_SOURCE_INCLUDE
}

runtime_verify_prepare_build_dir() {
  if [ -f build/CMakeCache.txt ]; then
    local cache_source_dir
    cache_source_dir="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt)"
    if [ -n "${cache_source_dir}" ] && [ "${cache_source_dir}" != "${PROJECT_ROOT}" ]; then
      echo "Recreating build/ because CMake cache points to ${cache_source_dir}"
      rm -rf build
    fi
  fi

  if [ -f build/compile_commands.json ]; then
    if grep -Eq '/Library/Developer/CommandLineTools/SDKs/MacOSX\.sdk|-arch arm64' \
        build/compile_commands.json; then
      echo "Recreating build/ because compile_commands.json contains host-specific macOS toolchain paths"
      rm -rf build
    fi
  fi

  if cmake -S . -B build -DLLVM_BUILD_DIR="${LLVM_BUILD}" >/dev/null 2>&1; then
    return 0
  fi

  echo "Recreating build/ because CMake configure failed"
  rm -rf build
  cmake -S . -B build -DLLVM_BUILD_DIR="${LLVM_BUILD}" >/dev/null
}

runtime_verify_build_targets() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  cmake --build build --target "$@" -j2 >/dev/null
}

runtime_verify_build_runtime_core() {
  if [ "${RUNTIME_VERIFY_RUNTIME_CORE_READY:-0}" = "1" ]; then
    return 0
  fi
  runtime_verify_build_targets AscendCRuntime AFIRRuntimeCAPI runtime-session
  export RUNTIME_VERIFY_RUNTIME_CORE_READY=1
}

runtime_verify_build_example_toolchain() {
  if [ "${RUNTIME_VERIFY_EXAMPLE_TOOLCHAIN_READY:-0}" = "1" ]; then
    return 0
  fi
  if [ -x build/bin/afir-opt ] && [ -x build/bin/afir-translate ]; then
    export RUNTIME_VERIFY_EXAMPLE_TOOLCHAIN_READY=1
    return 0
  fi
  runtime_verify_build_targets afir-opt afir-translate
  export RUNTIME_VERIFY_EXAMPLE_TOOLCHAIN_READY=1
}

runtime_verify_build_mix_compiler() {
  if [ "${RUNTIME_VERIFY_MIX_COMPILER_READY:-0}" = "1" ]; then
    return 0
  fi
  if [ -x build/bin/mix-compiler ]; then
    export RUNTIME_VERIFY_MIX_COMPILER_READY=1
    return 0
  fi
  runtime_verify_build_targets mix-compiler
  export RUNTIME_VERIFY_MIX_COMPILER_READY=1
}

runtime_verify_runtime_ld_library_path() {
  printf '%s\n' "${PROJECT_ROOT}/build/lib:${LLVM_BUILD}/lib:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_STUB_LIB}:${DEVICE_LIB64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

runtime_verify_cann_tiling_link_flags() {
  printf '%s\n' "-L${ASCEND_LIB64} -L${SOC_SIM_LIB} -L${DEVICE_LIB64} -ltiling_api -lregister -lplatform -lascendalog -lunified_dlog -lruntime_camodel -lnpu_drv -lstars -lmodel_top -lascendcl -lerror_manager -lprofapi -lge_common_base -lascend_dump -lmmpa -lc_sec"
}

runtime_verify_mix_ld_library_path() {
  local artifact_root="$1"
  printf '%s\n' "${artifact_root}/out:$(runtime_verify_runtime_ld_library_path)"
}
