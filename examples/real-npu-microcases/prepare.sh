#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
OUT_DIR=""
SKIP_COMPILE=false

usage() {
  cat <<'EOF' >&2
Usage:
  prepare.sh --out-dir <dir> [--skip-compile]

Generates real-NPU microcase data, kernels, run manifests, and optionally
runtime-session artifacts.
EOF
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir)
      [ "$#" -ge 2 ] || usage
      OUT_DIR="$2"
      shift 2
      ;;
    --skip-compile)
      SKIP_COMPILE=true
      shift
      ;;
    *)
      usage
      ;;
  esac
done

[ -n "${OUT_DIR}" ] || usage
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
"${PYTHON}" "${SCRIPT_DIR}/gen_data.py" --out-dir "${OUT_DIR}"

copy_kernel() {
  local case_name="$1"
  mkdir -p "${OUT_DIR}/${case_name}"
  cp "${SCRIPT_DIR}/kernels/${case_name}.cpp" "${OUT_DIR}/${case_name}/${case_name}.cpp"
}

write_manifest() {
  local case_name="$1"
  local inputs_json="$2"
  local shape_json="$3"
  local artifact_root="${OUT_DIR}/${case_name}/artifact"
  mkdir -p "${artifact_root}"
  cat >"${OUT_DIR}/${case_name}/run_manifest.json" <<EOF
{
  "backend": "npu",
  "artifact_root": "${artifact_root}",
  "tasks": [
    {
      "task_id": "${case_name}",
      "inputs": ${inputs_json},
      "outputs": [
        {
          "name": "out",
          "path": "${OUT_DIR}/${case_name}/output.npy",
          "shape": ${shape_json},
          "dtype": "f16"
        }
      ],
      "expected_outputs": [
        { "name": "out", "path": "${OUT_DIR}/${case_name}/expected.npy" }
      ],
      "block_dim": 1,
      "workspace_size": 8192,
      "profiling": true,
      "atol": 0.0,
      "rtol": 0.0
    }
  ]
}
EOF
}

compile_case() {
  local case_name="$1"
  if "${SKIP_COMPILE}"; then
    return
  fi
  "${RUNTIME_SESSION}" \
    --kernel "${OUT_DIR}/${case_name}/${case_name}.cpp" \
    --kernel-kind vec \
    --name "${case_name}" \
    --output "${OUT_DIR}/${case_name}/artifact"
}

copy_kernel const640
write_manifest const640 '[]' '[640]'
compile_case const640

copy_kernel copy640
write_manifest copy640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/copy640/input.npy" }]' \
  '[640]'
compile_case copy640

copy_kernel copy_tbuf640
write_manifest copy_tbuf640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/copy_tbuf640/input.npy" }]' \
  '[640]'
compile_case copy_tbuf640

copy_kernel copy_scalar640
write_manifest copy_scalar640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/copy_scalar640/input.npy" }]' \
  '[640]'
compile_case copy_scalar640

copy_kernel copy_params640
write_manifest copy_params640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/copy_params640/input.npy" }]' \
  '[640]'
compile_case copy_params640

copy_kernel copy_wait640
write_manifest copy_wait640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/copy_wait640/input.npy" }]' \
  '[640]'
compile_case copy_wait640

copy_kernel const_with_input640
write_manifest const_with_input640 \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/const_with_input640/input.npy" }]' \
  '[640]'
compile_case const_with_input640

copy_kernel relu_only
write_manifest relu_only \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/relu_only/input.npy" }]' \
  '[640]'
compile_case relu_only

copy_kernel broadcast_add
write_manifest broadcast_add \
  '[{ "name": "input", "path": "'"${OUT_DIR}"'/broadcast_add/input.npy" }, { "name": "bias", "path": "'"${OUT_DIR}"'/broadcast_add/bias.npy" }]' \
  '[2, 640]'
compile_case broadcast_add

echo "real NPU microcases prepared under ${OUT_DIR}"
