#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running SimBackend example tests"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"
source "${PROJECT_ROOT}/examples/env.sh" >/dev/null

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "${LLVM_BUILD}" ]; then
  exit 1
fi

if [ -f build/CMakeCache.txt ]; then
  CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt)"
  if [ -n "${CACHE_SOURCE_DIR}" ] && [ "${CACHE_SOURCE_DIR}" != "${PROJECT_ROOT}" ]; then
    rm -rf build
  fi
fi

cmake -S . -B build -DLLVM_BUILD_DIR="${LLVM_BUILD}" >/dev/null
cmake --build build \
  --target runtime-session afir-opt afir-translate mix-compiler \
  -j2 >/dev/null

RUNTIME_SESSION="${PROJECT_ROOT}/build/bin/runtime-session"
SELECTED_EXAMPLES=("$@")

ARTIFACT_ROOTS=()
MANIFESTS=()
OUTPUTS=()
SUMMARY_NAMES=()
SUMMARY_KINDS=()
SUMMARY_RETRIES=()
SUMMARY_OUTPUTS=()
SUMMARY_PROFILES=()
EXAMPLE_LOGS=()
CURRENT_RETRIES=0
LAST_PROFILE_PATH=""

cleanup() {
  rm -rf "${ARTIFACT_ROOTS[@]:-}"
  rm -f "${MANIFESTS[@]:-}" "${OUTPUTS[@]:-}" "${EXAMPLE_LOGS[@]:-}"
}
trap cleanup EXIT

make_tmp_artifact_root() {
  local dir
  dir="$(mktemp -d /tmp/runtime-sim-artifact.XXXXXX)"
  ARTIFACT_ROOTS+=("${dir}")
  printf '%s\n' "${dir}"
}

make_tmp_manifest() {
  local path
  path="$(mktemp /tmp/runtime-sim-manifest.XXXXXX.json)"
  MANIFESTS+=("${path}")
  printf '%s\n' "${path}"
}

make_tmp_output() {
  local path
  path="$(mktemp /tmp/runtime-sim-output.XXXXXX.npy)"
  OUTPUTS+=("${path}")
  printf '%s\n' "${path}"
}

make_tmp_log() {
  local path
  path="$(mktemp /tmp/runtime-sim-example.XXXXXX.log)"
  EXAMPLE_LOGS+=("${path}")
  printf '%s\n' "${path}"
}

compare_npy() {
  local expected="$1"
  local actual="$2"
  local atol="$3"
  local rtol="$4"
  python3 - "$expected" "$actual" "$atol" "$rtol" <<'PY'
import sys
import numpy as np

expected = np.load(sys.argv[1])
actual = np.load(sys.argv[2])
atol = float(sys.argv[3])
rtol = float(sys.argv[4])

if expected.shape != actual.shape:
    raise SystemExit(f"shape mismatch: {actual.shape} vs {expected.shape}")
if expected.dtype != actual.dtype:
    raise SystemExit(f"dtype mismatch: {actual.dtype} vs {expected.dtype}")
if not np.allclose(actual, expected, atol=atol, rtol=rtol):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    raise SystemExit(
        f"allclose failed: max_abs_diff={diff.max():.6e} mean_abs_diff={diff.mean():.6e}"
    )
print(f"compare ok: shape={actual.shape} dtype={actual.dtype}")
PY
}

run_runtime_session_manifest() {
  local manifest="$1"
  local status=0
  CURRENT_RETRIES=0
  LAST_PROFILE_PATH=""
  if "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run >/tmp/runtime_simbackend_run.log 2>&1; then
    LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' /tmp/runtime_simbackend_run.log | head -n1)"
    return 0
  fi
  status=$?
  if [ "${status}" -ne 134 ] && [ "${status}" -ne 139 ]; then
    return "${status}"
  fi
  echo "retrying runtime-session after simulator process exit ${status}" >&2
  sleep 1
  CURRENT_RETRIES=1
  "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run >/tmp/runtime_simbackend_run.log 2>&1
  LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' /tmp/runtime_simbackend_run.log | head -n1)"
}

record_summary() {
  local name="$1"
  local kind="$2"
  local retries="$3"
  local output_path="$4"
  local profile_path="$5"
  SUMMARY_NAMES+=("${name}")
  SUMMARY_KINDS+=("${kind}")
  SUMMARY_RETRIES+=("${retries}")
  SUMMARY_OUTPUTS+=("${output_path}")
  SUMMARY_PROFILES+=("${profile_path}")
}

print_summary() {
  local i
  echo "--- SimBackend summary ---"
  for ((i = 0; i < ${#SUMMARY_NAMES[@]}; ++i)); do
    printf 'example=%s kind=%s status=pass retries=%s output=%s profile=%s\n' \
      "${SUMMARY_NAMES[$i]}" \
      "${SUMMARY_KINDS[$i]}" \
      "${SUMMARY_RETRIES[$i]}" \
      "${SUMMARY_OUTPUTS[$i]}" \
      "${SUMMARY_PROFILES[$i]}"
  done
}

should_run_example() {
  local name="$1"
  if [ "${#SELECTED_EXAMPLES[@]}" -eq 0 ]; then
    return 0
  fi
  local selected
  for selected in "${SELECTED_EXAMPLES[@]}"; do
    if [ "${selected}" = "${name}" ]; then
      return 0
    fi
  done
  return 1
}

run_vec_example() {
  local example_dir="$1"
  local kernel_name="$2"
  local inputs_json="$3"
  local expected_path="$4"
  local tiling_params="$5"
  local block_dim="$6"
  local atol="$7"
  local rtol="$8"
  local kernel_file="${9:-step8_kernel.cpp}"

  local artifact_root
  artifact_root="$(make_tmp_artifact_root)"
  local example_log
  example_log="$(make_tmp_log)"
  local manifest
  manifest="$(make_tmp_manifest)"
  local actual_output
  actual_output="$(make_tmp_output)"

  bash "${example_dir}/run.sh" 2>&1 | tee "${example_log}"
  grep -q '^session.backend=sim$' "${example_log}"
  grep -q '^session.result=success$' "${example_log}"
  grep -q '^session.validation=pass$' "${example_log}"

  "${RUNTIME_SESSION}" \
    --kernel "${example_dir}/${kernel_file}" \
    --kernel-kind vec \
    --name "${kernel_name}" \
    --output "${artifact_root}" >/tmp/runtime_simbackend_compile.log 2>&1

  cat > "${manifest}" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${artifact_root}",
  "inputs": [
${inputs_json}
  ],
  "outputs": [
    { "name": "out", "path": "${actual_output}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${expected_path}" }
  ],
  "tiling": {
    "schema": "${example_dir}/tiling_space.json",
    "params": "${tiling_params}"
  },
  "block_dim": ${block_dim},
  "workspace_size": 16777216,
  "profiling": true,
  "atol": ${atol},
  "rtol": ${rtol}
}
EOF

  echo "--- SimBackend vec example: $(basename "${example_dir}") ---"
  run_runtime_session_manifest "${manifest}"
  test -f "${actual_output}"
  compare_npy "${expected_path}" "${actual_output}" "${atol}" "${rtol}"
  record_summary \
    "$(basename "${example_dir}")" \
    "vec" \
    "${CURRENT_RETRIES}" \
    "${actual_output}" \
    "${LAST_PROFILE_PATH:-<none>}"
}

run_mix_example() {
  local example_dir="$1"
  local kernel_name="$2"
  local data_dir="$3"
  local inputs_json="$4"
  local expected_path="$5"
  local atol="$6"
  local rtol="$7"

  local artifact_root
  artifact_root="$(make_tmp_artifact_root)"
  local example_log
  example_log="$(make_tmp_log)"
  local manifest
  manifest="$(make_tmp_manifest)"
  local actual_output
  actual_output="$(make_tmp_output)"

  bash "${example_dir}/run.sh" 2>&1 | tee "${example_log}"
  grep -q '^session.backend=sim$' "${example_log}"
  grep -q '^session.result=success$' "${example_log}"

  local dav_sim_version="${ASCEND_DAV_SIM_VERSION:-dav_3002}"
  local cann_arch
  cann_arch="$(resolve_cann_arch_dir)"
  local ascend_lib64="${ASCEND_HOME_PATH}/${cann_arch}/lib64"
  local soc_sim_lib="${ASCEND_HOME_PATH}/${cann_arch}/simulator/Ascend910B1/lib"
  local dav_sim_lib="${ASCEND_HOME_PATH}/${cann_arch}/simulator/${dav_sim_version}/lib"
  local device_lib="${ASCEND_HOME_PATH}/${cann_arch}/lib64/device/lib64"
  ASCEND_DAV_SIM_VERSION="${dav_sim_version}" "${RUNTIME_SESSION}" \
    --kernel "${example_dir}/step8_kernel.cpp" \
    --kernel-kind mix \
    --name "${kernel_name}" \
    --cann-mlir "${example_dir}/step7_cann.mlir" \
    --npy-dir "${data_dir}" \
    --output "${artifact_root}" >/tmp/runtime_simbackend_mix_compile.log 2>&1

  cat > "${manifest}" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${artifact_root}",
  "inputs": [
${inputs_json}
  ],
  "outputs": [
    { "name": "out", "path": "${actual_output}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${expected_path}" }
  ],
  "tiling": {
    "binary": "${artifact_root}/out/tiling.bin"
  },
  "block_dim": 1,
  "workspace_size": 16777216,
  "profiling": true,
  "atol": ${atol},
  "rtol": ${rtol}
}
EOF

  echo "--- SimBackend mix example: $(basename "${example_dir}") ---"
  local status=0
  CURRENT_RETRIES=0
  LAST_PROFILE_PATH=""
  if ! ASCEND_DAV_SIM_VERSION="${dav_sim_version}" \
    LD_LIBRARY_PATH="${artifact_root}/out:${ascend_lib64}:${soc_sim_lib}:${dav_sim_lib}:${device_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run \
    >/tmp/runtime_simbackend_run.log 2>&1; then
    status=$?
    if [ "${status}" -ne 134 ] && [ "${status}" -ne 139 ]; then
      return "${status}"
    fi
    echo "retrying runtime-session mix run after simulator process exit ${status}" >&2
    sleep 1
    CURRENT_RETRIES=1
    ASCEND_DAV_SIM_VERSION="${dav_sim_version}" \
    LD_LIBRARY_PATH="${artifact_root}/out:${ascend_lib64}:${soc_sim_lib}:${dav_sim_lib}:${device_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run \
      >/tmp/runtime_simbackend_run.log 2>&1
  fi
  LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' /tmp/runtime_simbackend_run.log | head -n1)"
  test -f "${actual_output}"
  compare_npy "${expected_path}" "${actual_output}" "${atol}" "${rtol}"
  record_summary \
    "$(basename "${example_dir}")" \
    "mix" \
    "${CURRENT_RETRIES}" \
    "${actual_output}" \
    "${LAST_PROFILE_PATH:-<none>}"
}

if should_run_example "relu-broadcast-transpose"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/relu-broadcast-transpose" \
  "relu_transpose_broadcast_add" \
  "    { \"name\": \"data0\", \"path\": \"${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data0.npy\" },
    { \"name\": \"data1\", \"path\": \"${PROJECT_ROOT}/examples/relu-broadcast-transpose/input_data1.npy\" }" \
  "${PROJECT_ROOT}/examples/relu-broadcast-transpose/output_expected.npy" \
  "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640" \
  "8" "1e-2" "1e-2"
fi

if should_run_example "add-broadcast-concat"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/add-broadcast-concat" \
  "ewop_broadcast_concat" \
  "    { \"name\": \"a\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/input_a.npy\" },
    { \"name\": \"b\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/input_b.npy\" },
    { \"name\": \"c\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/input_c.npy\" },
    { \"name\": \"d\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/input_d.npy\" }" \
  "${PROJECT_ROOT}/examples/add-broadcast-concat/output.npy" \
  "TB_M=64,TB_N=192,dim_arg0_0=640,dim_arg1_1=500,dim_arg2_0=640,dim_arg3_1=500,dim_arg0_1=500,dim_arg1_0=640,dim_arg2_1=500,dim_arg3_0=640" \
  "10" "1e-3" "1e-3"
fi

if should_run_example "broadcast-add-reduce"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/broadcast-add-reduce" \
  "broadcast_add_reducesum" \
  "    { \"name\": \"a\", \"path\": \"${PROJECT_ROOT}/examples/broadcast-add-reduce/input_a.npy\" },
    { \"name\": \"b\", \"path\": \"${PROJECT_ROOT}/examples/broadcast-add-reduce/input_b.npy\" }" \
  "${PROJECT_ROOT}/examples/broadcast-add-reduce/output_c.npy" \
  "TB_M=64,TB_N=15000,dim_arg0_0=640,dim_arg1_1=15000,dim_arg0_1=640,dim_arg1_0=15000" \
  "10" "10" "1e-2"
fi

if should_run_example "gather-elementwise-fusion"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/gather-elementwise-fusion" \
  "relu_index_select_add" \
  "    { \"name\": \"data\", \"path\": \"${PROJECT_ROOT}/examples/gather-elementwise-fusion/input_data.npy\" },
    { \"name\": \"indices\", \"path\": \"${PROJECT_ROOT}/examples/gather-elementwise-fusion/input_indices.npy\" },
    { \"name\": \"bias\", \"path\": \"${PROJECT_ROOT}/examples/gather-elementwise-fusion/input_bias.npy\" }" \
  "${PROJECT_ROOT}/examples/gather-elementwise-fusion/output_out.npy" \
  "TB_M=64,TB_N=1,dim_arg0_0=512,dim_arg1_0=256,dim_arg0_1=640,dim_arg1_1=256" \
  "8" "10" "1e-2" "step8_kernel_gen.cpp"
fi

if should_run_example "split-relu-brc-add-mul"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/split-relu-brc-add-mul" \
  "ewop_broadcast_split" \
  "    { \"name\": \"input_a\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/input_a.npy\" },
    { \"name\": \"bias0\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/bias0.npy\" },
    { \"name\": \"bias1\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/bias1.npy\" },
    { \"name\": \"scale0\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/scale0.npy\" },
    { \"name\": \"scale1\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/scale1.npy\" }" \
  "${PROJECT_ROOT}/examples/split-relu-brc-add-mul/output.npy" \
  "TB_M=16,TB_N=16,dim_arg0_1=512,dim_arg1_0=320,dim_arg0_0=640,dim_arg1_1=320,dim_arg3_0=512,dim_arg3_1=512,dim_arg2_0=320,dim_arg2_1=320,dim_arg4_0=512,dim_arg4_1=512" \
  "20" "1e-2" "1e-2"
fi

if should_run_example "matmul-add-leakyrelu"; then
run_mix_example \
  "${PROJECT_ROOT}/examples/matmul-add-leakyrelu" \
  "matmul_add_leakyrelu" \
  "${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/npy" \
  "    { \"name\": \"a\", \"path\": \"${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/npy/input_a.npy\" },
    { \"name\": \"b\", \"path\": \"${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/npy/input_b.npy\" },
    { \"name\": \"bias\", \"path\": \"${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/npy/input_bias.npy\" }" \
  "${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/npy/output.npy" \
  "1.0" "1e-2"
fi

print_summary
echo "SimBackend example baseline passed"
