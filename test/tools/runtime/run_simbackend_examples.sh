#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env
export LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)"
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

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
SUMMARY_SUMMARIES=()
EXAMPLE_LOGS=()
CURRENT_RETRIES=0
LAST_PROFILE_PATH=""
LAST_SUMMARY_PATH=""

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
  LAST_SUMMARY_PATH=""
  if "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run >/tmp/runtime_simbackend_run.log 2>&1; then
    LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' /tmp/runtime_simbackend_run.log | head -n1)"
    LAST_SUMMARY_PATH="$(sed -n 's/^session\.profile\.summary=//p' /tmp/runtime_simbackend_run.log | head -n1)"
    test -f "${LAST_SUMMARY_PATH}"
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
  LAST_SUMMARY_PATH="$(sed -n 's/^session\.profile\.summary=//p' /tmp/runtime_simbackend_run.log | head -n1)"
  test -f "${LAST_SUMMARY_PATH}"
}

record_summary() {
  local name="$1"
  local kind="$2"
  local retries="$3"
  local output_path="$4"
  local profile_path="$5"
  local summary_path="$6"
  SUMMARY_NAMES+=("${name}")
  SUMMARY_KINDS+=("${kind}")
  SUMMARY_RETRIES+=("${retries}")
  SUMMARY_OUTPUTS+=("${output_path}")
  SUMMARY_PROFILES+=("${profile_path}")
  SUMMARY_SUMMARIES+=("${summary_path}")
}

print_summary() {
  local i
  echo "--- SimBackend summary ---"
  for ((i = 0; i < ${#SUMMARY_NAMES[@]}; ++i)); do
    printf 'example=%s kind=%s status=pass retries=%s output=%s profile=%s summary=%s\n' \
      "${SUMMARY_NAMES[$i]}" \
      "${SUMMARY_KINDS[$i]}" \
      "${SUMMARY_RETRIES[$i]}" \
      "${SUMMARY_OUTPUTS[$i]}" \
      "${SUMMARY_PROFILES[$i]}" \
      "${SUMMARY_SUMMARIES[$i]}"
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
  local tiling_schema="${10:-${example_dir}/tiling_space.json}"

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
    --soc "${SOC_VERSION}" \
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
    "schema": "${tiling_schema}",
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
    "${LAST_PROFILE_PATH:-<none>}" \
    "${LAST_SUMMARY_PATH:-<none>}"
}

run_gather_mainline_shape() {
  local example_dir="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local block_dim="$5"

  local example_log
  example_log="$(make_tmp_log)"

  echo "--- SimBackend vec example: $(basename "${example_dir}") M=${m} N=${n} K=${k} block_dim=${block_dim} ---"
  bash "${example_dir}/run.sh" \
    --m "${m}" --n "${n}" --k "${k}" --block-dim "${block_dim}" --log \
    2>&1 | tee "${example_log}"
  grep -q '^session.backend=sim$' "${example_log}"
  grep -q '^session.result=success$' "${example_log}"
  grep -q '^session.validation=pass$' "${example_log}"

  local validation_log="${example_dir}/build_mainline/runtime_session.log"
  local actual_output="${example_dir}/build_mainline/output_actual.npy"
  local summary_output
  summary_output="$(make_tmp_output)"
  test -f "${validation_log}"
  test -f "${actual_output}"
  cp "${actual_output}" "${summary_output}"
  LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' "${validation_log}" | head -n1)"
  LAST_SUMMARY_PATH="$(sed -n 's/^session\.profile\.summary=//p' "${validation_log}" | head -n1)"
  test -f "${LAST_SUMMARY_PATH}"
  record_summary \
    "$(basename "${example_dir}")-M${m}-N${n}-K${k}" \
    "vec" \
    "0" \
    "${summary_output}" \
    "${LAST_PROFILE_PATH:-<none>}" \
    "${LAST_SUMMARY_PATH:-<none>}"
}

run_mix_example() {
  local example_dir="$1"
  local kernel_name="$2"
  local data_dir="$3"
  local inputs_json="$4"
  local expected_path="$5"
  local atol="$6"
  local rtol="$7"
  local kernel_file="${8:-step8_kernel.cpp}"
  local cann_mlir="${9:-step7_cann.mlir}"

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

  ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" "${RUNTIME_SESSION}" \
    --kernel "${example_dir}/${kernel_file}" \
    --kernel-kind mix \
    --name "${kernel_name}" \
    --cann-mlir "${example_dir}/${cann_mlir}" \
    --npy-dir "${data_dir}" \
    --soc "${SOC_VERSION}" \
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
  if ! ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
    LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${artifact_root}")" \
    "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run \
    >/tmp/runtime_simbackend_run.log 2>&1; then
    status=$?
    if [ "${status}" -ne 134 ] && [ "${status}" -ne 139 ]; then
      return "${status}"
    fi
    echo "retrying runtime-session mix run after simulator process exit ${status}" >&2
    sleep 1
    CURRENT_RETRIES=1
    ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
    LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${artifact_root}")" \
      "${RUNTIME_SESSION}" --run-manifest "${manifest}" --run \
      >/tmp/runtime_simbackend_run.log 2>&1
  fi
  LAST_PROFILE_PATH="$(sed -n 's/^session\.profile\[[0-9][0-9]*\]=//p' /tmp/runtime_simbackend_run.log | head -n1)"
  LAST_SUMMARY_PATH="$(sed -n 's/^session\.profile\.summary=//p' /tmp/runtime_simbackend_run.log | head -n1)"
  test -f "${LAST_SUMMARY_PATH}"
  test -f "${actual_output}"
  compare_npy "${expected_path}" "${actual_output}" "${atol}" "${rtol}"
  record_summary \
    "$(basename "${example_dir}")" \
    "mix" \
    "${CURRENT_RETRIES}" \
    "${actual_output}" \
    "${LAST_PROFILE_PATH:-<none>}" \
    "${LAST_SUMMARY_PATH:-<none>}"
}

if should_run_example "relu-broadcast-transpose"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/relu-broadcast-transpose" \
  "relu_transpose_broadcast_add" \
  "    { \"name\": \"data0\", \"path\": \"${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data0.npy\" },
    { \"name\": \"data1\", \"path\": \"${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data1.npy\" }" \
  "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/output_expected.npy" \
  "dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640" \
  "20" "1e-2" "1e-2" \
  "build_mainline/step10_kernel.cpp" \
  "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/phase5_tiling_space.json"
fi

if should_run_example "add-broadcast-concat"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/add-broadcast-concat" \
  "ewop_broadcast_concat" \
  "    { \"name\": \"arg0\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/input_a.npy\" },
    { \"name\": \"arg1\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/input_b.npy\" },
    { \"name\": \"arg2\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/input_c.npy\" },
    { \"name\": \"arg3\", \"path\": \"${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/input_d.npy\" }" \
  "${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/output.npy" \
  "dim_arg0_0=640,dim_arg1_1=500,dim_arg1_0=640,dim_arg2_0=640,dim_arg3_0=640,dim_arg3_1=500" \
  "20" "1e-2" "1e-2" \
  "build_mainline/step10_kernel.cpp" \
  "${PROJECT_ROOT}/examples/add-broadcast-concat/build_mainline/phase5_tiling_space.json"
fi

if should_run_example "broadcast-add-reduce"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/broadcast-add-reduce" \
  "broadcast_add_reducesum" \
  "    { \"name\": \"a\", \"path\": \"${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/input_a.npy\" },
    { \"name\": \"b\", \"path\": \"${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/input_b.npy\" }" \
  "${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/output_c.npy" \
  "dim_arg0_0=640,dim_arg1_1=15000,dim_arg1_0=640" \
  "20" "10" "1e-2" \
  "build_mainline/step10_kernel.cpp" \
  "${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/phase5_tiling_space.json"
fi

if should_run_example "gather-elementwise-fusion"; then
run_gather_mainline_shape "${PROJECT_ROOT}/examples/gather-elementwise-fusion" 65 127 31 20
run_gather_mainline_shape "${PROJECT_ROOT}/examples/gather-elementwise-fusion" 96 128 31 20
run_gather_mainline_shape "${PROJECT_ROOT}/examples/gather-elementwise-fusion" 96 127 32 20
fi

if should_run_example "split-relu-brc-add-mul"; then
run_vec_example \
  "${PROJECT_ROOT}/examples/split-relu-brc-add-mul" \
  "ewop_broadcast_split" \
  "    { \"name\": \"input_a\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/input_a.npy\" },
    { \"name\": \"bias0\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/bias0.npy\" },
    { \"name\": \"bias1\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/bias1.npy\" },
    { \"name\": \"scale0\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/scale0.npy\" },
    { \"name\": \"scale1\", \"path\": \"${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/scale1.npy\" }" \
  "${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/output.npy" \
  "dim_arg0_1=512,dim_arg1_0=320,dim_arg0_0=640,dim_arg3_0=512,dim_arg2_0=320,dim_arg4_0=512" \
  "20" "1e-2" "1e-2" \
  "build_mainline/step10_kernel.cpp" \
  "${PROJECT_ROOT}/examples/split-relu-brc-add-mul/build_mainline/phase5_tiling_space.json"
fi

if should_run_example "matmul-add-leakyrelu"; then
run_mix_example \
  "${PROJECT_ROOT}/examples/matmul-add-leakyrelu" \
  "matmul_add_leakyrelu" \
  "${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/npy" \
  "    { \"name\": \"a\", \"path\": \"${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/npy/input_a.npy\" },
    { \"name\": \"b\", \"path\": \"${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/npy/input_b.npy\" },
    { \"name\": \"bias\", \"path\": \"${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/npy/input_bias.npy\" }" \
  "${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/npy/output.npy" \
  "1.0" "1e-2" \
  "build_mainline/step10_kernel.cpp" \
  "build_mainline/step8_cann.mlir"
fi

print_summary
echo "SimBackend example baseline passed"
