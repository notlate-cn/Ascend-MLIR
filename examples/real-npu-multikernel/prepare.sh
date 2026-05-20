#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROCASE_DIR="$(cd "${SCRIPT_DIR}/../real-npu-microcases" && pwd)"
PYTHON="${PYTHON:-python3}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
OUT_DIR=""
SKIP_COMPILE=false

usage() {
  cat <<'EOF' >&2
Usage:
  prepare.sh --out-dir <dir> [--skip-compile]

Generates real-NPU multi-kernel scheduling data, kernels, run manifests, and
optionally runtime-session artifacts.
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
  local kernel_name="$1"
  local src="$2"
  local kernel_dir="${OUT_DIR}/artifacts/${kernel_name}"
  mkdir -p "${kernel_dir}/artifact"
  cp "${src}" "${kernel_dir}/${kernel_name}.cpp"
}

compile_kernel() {
  local kernel_name="$1"
  if "${SKIP_COMPILE}"; then
    return
  fi
  "${RUNTIME_SESSION}" \
    --kernel "${OUT_DIR}/artifacts/${kernel_name}/${kernel_name}.cpp" \
    --kernel-kind vec \
    --name "${kernel_name}" \
    --output "${OUT_DIR}/artifacts/${kernel_name}/artifact"
}

common_task_fields() {
  cat <<'EOF'
      "block_dim": 1,
      "workspace_size": 8192,
      "profiling": true,
      "atol": 0.0,
      "rtol": 0.0
EOF
}

write_serial_manifest() {
  mkdir -p "${OUT_DIR}/serial-two-kernel"
  cat >"${OUT_DIR}/serial-two-kernel/run_manifest.json" <<EOF
{
  "backend": "npu",
  "tasks": [
    {
      "task_id": "producer",
      "artifact_root": "${OUT_DIR}/artifacts/const640/artifact",
      "outputs": [
        { "name": "mid", "shape": [640], "dtype": "f16" }
      ],
$(common_task_fields)
    },
    {
      "task_id": "consumer",
      "artifact_root": "${OUT_DIR}/artifacts/copy_scalar640/artifact",
      "dependencies": ["producer"],
      "inputs": [
        {
          "name": "input",
          "source": "task_output",
          "upstream_task": "producer",
          "upstream_output": "mid"
        }
      ],
      "outputs": [
        {
          "name": "out",
          "path": "${OUT_DIR}/serial-two-kernel/output.npy",
          "shape": [640],
          "dtype": "f16"
        }
      ],
      "expected_outputs": [
        { "name": "out", "path": "${OUT_DIR}/serial-two-kernel/expected.npy" }
      ],
$(common_task_fields)
    }
  ]
}
EOF
}

write_fork_join_manifest() {
  mkdir -p "${OUT_DIR}/fork-join"
  cat >"${OUT_DIR}/fork-join/run_manifest.json" <<EOF
{
  "backend": "npu",
  "tasks": [
    {
      "task_id": "producer_a",
      "artifact_root": "${OUT_DIR}/artifacts/const640/artifact",
      "outputs": [
        { "name": "a", "shape": [640], "dtype": "f16" }
      ],
$(common_task_fields)
    },
    {
      "task_id": "producer_b",
      "artifact_root": "${OUT_DIR}/artifacts/const_with_input640/artifact",
      "inputs": [
        { "name": "input", "path": "${OUT_DIR}/data/producer_b/input.npy" }
      ],
      "outputs": [
        { "name": "b", "shape": [640], "dtype": "f16" }
      ],
$(common_task_fields)
    },
    {
      "task_id": "consumer",
      "artifact_root": "${OUT_DIR}/artifacts/add_wait640/artifact",
      "dependencies": ["producer_a", "producer_b"],
      "inputs": [
        {
          "name": "lhs",
          "source": "task_output",
          "upstream_task": "producer_a",
          "upstream_output": "a"
        },
        {
          "name": "rhs",
          "source": "task_output",
          "upstream_task": "producer_b",
          "upstream_output": "b"
        }
      ],
      "outputs": [
        {
          "name": "out",
          "path": "${OUT_DIR}/fork-join/output.npy",
          "shape": [640],
          "dtype": "f16"
        }
      ],
      "expected_outputs": [
        { "name": "out", "path": "${OUT_DIR}/fork-join/expected.npy" }
      ],
$(common_task_fields)
    }
  ]
}
EOF
}

copy_kernel const640 "${MICROCASE_DIR}/kernels/const640.cpp"
copy_kernel copy_scalar640 "${MICROCASE_DIR}/kernels/copy_scalar640.cpp"
copy_kernel const_with_input640 "${MICROCASE_DIR}/kernels/const_with_input640.cpp"
copy_kernel add_wait640 "${SCRIPT_DIR}/kernels/add_wait640.cpp"

compile_kernel const640
compile_kernel copy_scalar640
compile_kernel const_with_input640
compile_kernel add_wait640

write_serial_manifest
write_fork_join_manifest

echo "real NPU multi-kernel cases prepared under ${OUT_DIR}"
