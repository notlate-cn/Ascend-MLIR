#!/usr/bin/env bash
# test/tools/runtime/run_runtime.sh
# Builds the project and runs the task-graph runtime verification flow.
# Does NOT require a simulator or .bin file.
#
# Usage:
#   cd /path/to/Ascend-MLIR
#   export LLVM_BUILD_DIR=/path/to/llvm/build
#   bash test/tools/runtime/run_runtime.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env
export LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)"
runtime_verify_prepare_build_dir

echo "--- Building focused runtime verification targets ---"
runtime_verify_build_runtime_core

echo "--- Checking runtime-session CLI ---"
test -x build/bin/runtime-session
build/bin/runtime-session --help | grep -q "task graph runtime"

echo "--- Checking runtime execution runner API surface ---"
if grep -R -n -E "runPackedMixFile|PackedMixExecutionLaunch" \
    include/Runtime lib/Runtime test/tools/runtime/test_*.cpp; then
  echo "Error: packed mix runner API must not remain in active runtime surfaces" >&2
  exit 1
fi
if grep -R -n "packedSharedObjectPath" \
    include/Runtime/Execution lib/Runtime/Execution lib/CAPI/Runtime; then
  echo "Error: execution layer must consume generic shared-library artifact paths" >&2
  exit 1
fi
if grep -R -n "aclrtlaunch_" lib/Runtime/Execution; then
  echo "Error: execution layer must consume artifact-provided launch symbols" >&2
  exit 1
fi
if grep -R -n "configurePackedMixEnvironment" include/Runtime lib/Runtime; then
  echo "Error: packed mix simulator env helper must not remain" >&2
  exit 1
fi
if grep -R -n -E \
    "ASCEND_MIX_CONTRACT_MODE|AFIR_MIX_USE_LEGACY_RUNNER|legacy-preprocess|legacy_runner|build_legacy_tiling_runner" \
    include/Runtime lib/Runtime tools test/tools/runtime/test_*.cpp; then
  echo "Error: legacy mix compile switches must not remain in active runtime surfaces" >&2
  exit 1
fi

FAKE_ARTIFACT_ROOT="$(mktemp -d)"
INVALID_STDERR=""
CONFLICT_STDERR=""
INVALID_KIND_STDERR=""
RUN_STDERR=""
TEST_RUNTIME_BIN="$(mktemp /tmp/test_runtime.XXXXXX)"
TEST_TASKGRAPH_RUNTIME_BIN="$(mktemp /tmp/test_taskgraph_runtime.XXXXXX)"
TEST_CAPI_RUNTIME_BIN="$(mktemp /tmp/test_capi_runtime.XXXXXX)"
RUNTIME_SESSION_ARTIFACT_ROOT="$(mktemp -d)"
RUNTIME_SESSION_SECOND_ARTIFACT_ROOT="$(mktemp -d)"
RUNTIME_SESSION_RUN_MANIFEST="$(mktemp /tmp/runtime_session_run_manifest.XXXXXX.json)"
RUNTIME_SESSION_ACTUAL_OUTPUT="$(mktemp /tmp/runtime_session_actual.XXXXXX.npy)"
RUNTIME_SESSION_ARTIFACT_MANIFEST="$(mktemp /tmp/runtime_session_artifact_manifest.XXXXXX.json)"
RUNTIME_SESSION_PREPARED_RUN_MANIFEST="$(mktemp /tmp/runtime_session_prepared_run_manifest.XXXXXX.json)"
RUNTIME_SESSION_PREPARED_INPUT="$(mktemp /tmp/runtime_session_prepared_input.XXXXXX.npy)"
RUNTIME_SESSION_PREPARED_OUTPUT="$(mktemp /tmp/runtime_session_prepared_output.XXXXXX.npy)"
RUNTIME_SESSION_PREPARED_EXPECTED="$(mktemp /tmp/runtime_session_prepared_expected.XXXXXX.npy)"
RUNTIME_SESSION_PREPARE_BINDING_STDERR="$(mktemp /tmp/runtime_session_prepare_binding.XXXXXX.err)"
RUNTIME_SESSION_TENSOR_DIFF_DIR="$(mktemp -d)"
RUNTIME_SESSION_CASE_DIR="$(mktemp -d)"
RUNTIME_SESSION_CASE_JSON="${RUNTIME_SESSION_CASE_DIR}/case.json"
RUNTIME_SESSION_CASE_RUN_MANIFEST="${RUNTIME_SESSION_CASE_DIR}/run_manifest.json"
RUNTIME_SESSION_CASE_CONFLICT_STDERR="${RUNTIME_SESSION_CASE_DIR}/conflict.err"
RUNTIME_SESSION_DAG_MANIFEST="$(mktemp /tmp/runtime_session_dag_manifest.XXXXXX.json)"
RUNTIME_SESSION_DAG_OUTPUT="$(mktemp /tmp/runtime_session_dag_actual.XXXXXX.npy)"
RUNTIME_SESSION_NPU_MANIFEST="$(mktemp /tmp/runtime_session_npu_manifest.XXXXXX.json)"
RUNTIME_SESSION_NPU_OUTPUT="$(mktemp /tmp/runtime_session_npu_actual.XXXXXX.npy)"
RUNTIME_SESSION_NPU_SUCCESS_STDOUT="$(mktemp /tmp/runtime_session_npu_success.XXXXXX.log)"
RUNTIME_SESSION_NPU_SUCCESS_STDERR="$(mktemp /tmp/runtime_session_npu_success_stderr.XXXXXX.log)"
NPU_STDERR=""
cleanup() {
  rm -rf "$FAKE_ARTIFACT_ROOT"
  rm -rf "$RUNTIME_SESSION_ARTIFACT_ROOT"
  rm -rf "$RUNTIME_SESSION_SECOND_ARTIFACT_ROOT"
  rm -rf "$RUNTIME_SESSION_TENSOR_DIFF_DIR"
  rm -rf "$RUNTIME_SESSION_CASE_DIR"
  rm -f "$INVALID_STDERR" "$RUN_STDERR" "$TEST_RUNTIME_BIN" \
        "$CONFLICT_STDERR" "$INVALID_KIND_STDERR" \
        "$TEST_TASKGRAPH_RUNTIME_BIN" "$TEST_CAPI_RUNTIME_BIN" \
        "$RUNTIME_SESSION_RUN_MANIFEST" \
        "$RUNTIME_SESSION_ACTUAL_OUTPUT" "$RUNTIME_SESSION_ARTIFACT_MANIFEST" \
        "$RUNTIME_SESSION_PREPARED_RUN_MANIFEST" \
        "$RUNTIME_SESSION_PREPARED_INPUT" \
        "$RUNTIME_SESSION_PREPARED_OUTPUT" \
        "$RUNTIME_SESSION_PREPARED_EXPECTED" \
        "$RUNTIME_SESSION_PREPARE_BINDING_STDERR" \
        "$RUNTIME_SESSION_DAG_MANIFEST" \
        "$RUNTIME_SESSION_DAG_OUTPUT" "$RUNTIME_SESSION_NPU_MANIFEST" \
        "$RUNTIME_SESSION_NPU_OUTPUT" "$RUNTIME_SESSION_NPU_SUCCESS_STDOUT" \
        "$RUNTIME_SESSION_NPU_SUCCESS_STDERR" "$NPU_STDERR"
}
trap cleanup EXIT
mkdir -p "${FAKE_ARTIFACT_ROOT}/out"
cat > "${FAKE_ARTIFACT_ROOT}/out/manifest.txt" <<'EOF'
kernel_name=fake_kernel
soc_version=Ascend910B1
kernel_kind=vec
device_binary_path=fake.bin
EOF

echo "--- Checking runtime-session planning path ---"
PLAN_OUTPUT="$(build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}")"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\[0\]=main"

echo "--- Checking runtime-session negative paths ---"
INVALID_STDERR="$(mktemp)"
RUN_STDERR="$(mktemp)"
if build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}/missing" 2>"${INVALID_STDERR}"; then
  echo "Error: invalid artifact root unexpectedly succeeded" >&2
  exit 1
fi
grep -q "cannot access artifact root" "${INVALID_STDERR}"

CONFLICT_STDERR="$(mktemp)"
if build/bin/runtime-session \
    --artifact-root "${FAKE_ARTIFACT_ROOT}" \
    --kernel examples/relu-broadcast-transpose/step0_input.mlir \
    2>"${CONFLICT_STDERR}"; then
  echo "Error: conflicting runtime-session inputs unexpectedly succeeded" >&2
  exit 1
fi
grep -q "provide exactly one of --artifact-root or --kernel" "${CONFLICT_STDERR}"

INVALID_KIND_STDERR="$(mktemp)"
if build/bin/runtime-session --kernel-kind invalid 2>"${INVALID_KIND_STDERR}"; then
  echo "Error: runtime-session invalid kernel-kind unexpectedly succeeded" >&2
  exit 1
fi
grep -q "provide exactly one of --artifact-root or --kernel" "${INVALID_KIND_STDERR}"

if build/bin/runtime-session --artifact-root "${FAKE_ARTIFACT_ROOT}" --run 2>"${RUN_STDERR}"; then
  echo "Error: runtime-session --run unexpectedly succeeded" >&2
  exit 1
fi
grep -q "simulation path requires at least one output binding" "${RUN_STDERR}"

echo "--- Checking runtime-session artifact-manifest prepare path ---"
cat > "${FAKE_ARTIFACT_ROOT}/host_tiling.cpp" <<'EOF'
#include <cstdint>
#include <cstring>

struct KernelATilingData {
  int64_t dim;
  int32_t tag;
  int32_t tile;
};

extern "C" int32_t kernel_a_GetTilingSize(void) {
  return static_cast<int32_t>(sizeof(KernelATilingData));
}

extern "C" int32_t kernel_a_GetTiling(const int64_t *shape_args,
                                      int32_t shape_count,
                                      void *tiling_out) {
  if (shape_count != 1 || shape_args == nullptr || tiling_out == nullptr)
    return 1;
  KernelATilingData data{shape_args[0], 1234, 32};
  std::memcpy(tiling_out, &data, sizeof(data));
  return 0;
}

extern "C" int64_t kernel_a_GetBlockDim(const int64_t *shape_args,
                                        int32_t shape_count) {
  if (shape_count != 1 || shape_args == nullptr)
    return -1;
  return shape_args[0] / 32;
}

extern "C" int64_t kernel_a_GetWorkspaceSize(const int64_t *shape_args,
                                             int32_t shape_count) {
  if (shape_count != 1 || shape_args == nullptr)
    return -1;
  return shape_args[0] * 3;
}
EOF
"${CXX:-c++}" -shared -fPIC "${FAKE_ARTIFACT_ROOT}/host_tiling.cpp" \
  -o "${FAKE_ARTIFACT_ROOT}/host_tiling.so"
for kernel in kernel_a kernel_b; do
  mkdir -p "${FAKE_ARTIFACT_ROOT}/${kernel}/out"
  cat > "${FAKE_ARTIFACT_ROOT}/${kernel}/out/manifest.txt" <<EOF
kernel_name=${kernel}
soc_version=Ascend910B1
kernel_kind=vec
device_binary_path=fake.bin
EOF
done
cat > "${RUNTIME_SESSION_ARTIFACT_MANIFEST}" <<'EOF'
{
  "hostTilingBindings": [
    {
      "id": "kernel_a_tiling",
      "library": "host_tiling.so",
      "symbols": {
        "getTilingSize": "kernel_a_GetTilingSize",
        "getTiling": "kernel_a_GetTiling",
        "getBlockDim": "kernel_a_GetBlockDim",
        "getWorkspaceSize": "kernel_a_GetWorkspaceSize"
      }
    }
  ],
  "kernelGraph": {
    "nodes": [
      { "name": "kernel_a" },
      { "name": "kernel_b" }
    ],
    "edges": [
      {
        "from": "kernel_a",
        "to": "kernel_b",
        "carriedBuffers": ["tmp0"]
      }
    ]
  },
  "kernel_entries": [
    {
      "kernel_id": "kernel_a",
      "kernelKind": "vec",
      "workspaceSizeBytes": 1024,
      "scheduleEntries": [
        {
          "decisionId": "kernel_a.decision.0",
          "guard": "arg0_dim0 % 32 == 0 && arg0_dim0 <= 256",
          "priority": 0,
          "hostTilingId": "kernel_a_tiling",
          "tilingParams": {
          }
        },
        {
          "decisionId": "kernel_a.fallback",
          "guard": "arg0_dim0 > 0",
          "priority": 99,
          "fallback": true,
          "tilingParams": {
          }
        }
      ],
      "shapeArgOrder": [
        { "name": "dim_arg0_0", "shapeKey": "arg0_dim0", "abiPosition": 0 }
      ],
      "abi": {
        "numInputs": 1,
        "numOutputs": 1,
        "workspaceArgIndex": 2,
        "inputs": [
          { "name": "arg0", "shape": [-1], "dtype": "f16" }
        ],
        "outputs": [
          { "name": "out0", "shape": [128], "dtype": "f16" }
        ]
      }
    },
    {
      "kernel_id": "kernel_b",
      "kernelKind": "vec",
      "workspaceSizeBytes": 2048,
      "scheduleEntries": [
        {
          "decisionId": "kernel_b.decision.0",
          "guard": "true",
          "tilingParams": {
          }
        }
      ],
      "abi": {
        "numInputs": 1,
        "numOutputs": 1,
        "workspaceArgIndex": 2,
        "inputs": [
          { "name": "arg0", "shape": [128], "dtype": "f16" }
        ],
        "outputs": [
          { "name": "out0", "shape": [128], "dtype": "f16" }
        ]
      }
    }
  ]
}
EOF
python3 - \
  "${RUNTIME_SESSION_PREPARED_INPUT}" \
  "${RUNTIME_SESSION_PREPARED_EXPECTED}" <<'PY'
import pathlib
import struct
import sys

def write_npy(path, element_count):
    header = "{'descr': '<f2', 'fortran_order': False, 'shape': (%d,), }" % element_count
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    payload = b"\x00\x00" * element_count
    pathlib.Path(path).write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload
    )

write_npy(sys.argv[1], 128)
write_npy(sys.argv[2], 128)
PY

echo "--- Checking runtime-session tensor comparison path ---"
mkdir -p "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/cpu" \
         "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/npu" \
         "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/summaries"
python3 - "${RUNTIME_SESSION_TENSOR_DIFF_DIR}" <<'PY'
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])

def write_f32(path, values):
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    payload = b"".join(struct.pack("<f", value) for value in values)
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload
    )

write_f32(root / "tensors/cpu/output0.npy", [1.0, 2.0, 3.0, 4.0])
write_f32(root / "tensors/npu/output0.npy", [1.0, 2.001, 3.0, 4.0])
write_f32(root / "tensors/npu/output_bad.npy", [1.0, 2.2, 3.0, 4.0])
PY
cat > "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/manifest.json" <<'JSON'
{
  "schema_version": 1,
  "comparisons": [
    {
      "id": "checkpoint/kernel_0",
      "kernel_id": "kernel_0",
      "task_id": "kernel_0",
      "lhs": "tensors/cpu/output0.npy",
      "rhs": "tensors/npu/output0.npy",
      "atol": 0.01,
      "rtol": 0.01
    }
  ]
}
JSON
build/bin/runtime-session \
  --compare-tensors "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/manifest.json" \
  --emit-validation-summary "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/summaries/tensor_diff.json" \
  >"${RUNTIME_SESSION_TENSOR_DIFF_DIR}/compare_pass.log"
grep -q '^validation.status=pass$' "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/compare_pass.log"
python3 - "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/summaries/tensor_diff.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary["schema_version"] == 1
assert summary["tool"] == "runtime-session"
assert summary["status"] == "pass"
assert summary["comparison_count"] == 1
assert summary["failed_count"] == 0
comparison = summary["comparisons"][0]
assert comparison["id"] == "checkpoint/kernel_0"
assert comparison["kernel_id"] == "kernel_0"
assert comparison["task_id"] == "kernel_0"
assert comparison["status"] == "pass"
assert comparison["shape"] == [4]
assert comparison["lhs_dtype"] == "f32"
assert comparison["rhs_dtype"] == "f32"
assert comparison["element_count"] == 4
assert comparison["max_abs_error"] > 0.0
assert comparison["max_rel_error"] > 0.0
assert comparison["mean_abs_error"] > 0.0
PY
cat > "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/manifest_fail.json" <<'JSON'
{
  "schema_version": 1,
  "comparisons": [
    {
      "id": "checkpoint/kernel_0",
      "kernel_id": "kernel_0",
      "task_id": "kernel_0",
      "lhs": "tensors/cpu/output0.npy",
      "rhs": "tensors/npu/output_bad.npy",
      "atol": 0.01,
      "rtol": 0.01
    }
  ]
}
JSON
if build/bin/runtime-session \
  --compare-tensors "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/tensors/manifest_fail.json" \
  --emit-validation-summary "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/summaries/tensor_diff_fail.json" \
  >"${RUNTIME_SESSION_TENSOR_DIFF_DIR}/compare_fail.log"; then
  echo "Error: runtime-session tensor comparison mismatch unexpectedly succeeded" >&2
  exit 1
fi
grep -q '^validation.status=fail$' "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/compare_fail.log"
python3 - "${RUNTIME_SESSION_TENSOR_DIFF_DIR}/summaries/tensor_diff_fail.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary["tool"] == "runtime-session"
assert summary["status"] == "fail"
assert summary["failed_count"] == 1
comparison = summary["comparisons"][0]
assert comparison["status"] == "fail"
assert comparison["max_abs_error"] > 0.1
assert comparison["max_rel_error"] > 0.05
PY

build/bin/runtime-session \
  --artifact-manifest "${RUNTIME_SESSION_ARTIFACT_MANIFEST}" \
  --artifact-root "${FAKE_ARTIFACT_ROOT}" \
  --shape-arg arg0_dim0=128 \
  --input "kernel_a.arg0=${RUNTIME_SESSION_PREPARED_INPUT}" \
  --output "kernel_b.out0=${RUNTIME_SESSION_PREPARED_OUTPUT}" \
  --expected-output "kernel_b.out0=${RUNTIME_SESSION_PREPARED_EXPECTED}" \
  --profiling \
  --atol 0.01 \
  --rtol 0.02 \
  --emit-run-manifest "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}" \
  >/tmp/runtime_session_prepare_manifest.log
test -f "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}"
python3 - \
  "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}" \
  "${FAKE_ARTIFACT_ROOT}" \
  "${RUNTIME_SESSION_PREPARED_INPUT}" \
  "${RUNTIME_SESSION_PREPARED_OUTPUT}" \
  "${RUNTIME_SESSION_PREPARED_EXPECTED}" <<'PY'
import json
import pathlib
import sys

def expect_equal(actual, expected, label):
    if actual != expected:
        raise SystemExit(f"{label}: expected {expected!r}, got {actual!r}")

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
artifact_root = sys.argv[2]
input_path = sys.argv[3]
output_path = sys.argv[4]
expected_path = sys.argv[5]
assert manifest["backend"] == "sim"
assert manifest["artifact_root"] == artifact_root
tasks = manifest["tasks"]
assert [task["task_id"] for task in tasks] == ["kernel_a", "kernel_b"]
assert tasks[0]["artifact_root"].endswith("/kernel_a")
assert tasks[1]["artifact_root"].endswith("/kernel_b")
assert tasks[0].get("dependencies", []) == []
assert tasks[1]["dependencies"] == ["kernel_a"]
assert tasks[0]["workspace_size"] == 384
assert tasks[1]["workspace_size"] == 2048
assert tasks[0]["block_dim"] == 4
tiling_path = pathlib.Path(tasks[0]["tiling"]["binary"])
assert tiling_path.exists(), tiling_path
assert tiling_path.stat().st_size == 16
expect_equal(tasks[0]["inputs"], [{
    "name": "arg0",
    "path": input_path,
    "shape": [128],
    "dtype": "f16",
}], "kernel_a inputs")
expect_equal(tasks[0]["outputs"], [{"name": "out0", "shape": [128], "dtype": "f16"}], "kernel_a outputs")
expect_equal(tasks[1]["inputs"], [{
    "name": "arg0",
    "source": "task_output",
    "upstream_task": "kernel_a",
    "upstream_output": "out0",
    "shape": [128],
    "dtype": "f16",
}], "kernel_b inputs")
expect_equal(tasks[1]["outputs"], [{
    "name": "out0",
    "path": output_path,
    "shape": [128],
    "dtype": "f16",
}], "kernel_b outputs")
expect_equal(tasks[1].get("expected_outputs"), [{
    "name": "out0",
    "path": expected_path,
    "shape": [128],
    "dtype": "f16",
}], "kernel_b expected_outputs")
expect_equal(tasks[1].get("profiling"), True, "kernel_b profiling")
expect_equal(tasks[1].get("atol"), 0.01, "kernel_b atol")
expect_equal(tasks[1].get("rtol"), 0.02, "kernel_b rtol")
PY
PLAN_OUTPUT="$(build/bin/runtime-session --run-manifest "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}")"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[0\\]=kernel_a"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[1\\]=kernel_b"
if build/bin/runtime-session \
  --artifact-manifest "${RUNTIME_SESSION_ARTIFACT_MANIFEST}" \
  --artifact-root "${FAKE_ARTIFACT_ROOT}" \
  --shape-arg arg0_dim0=128 \
  --input "missing=${RUNTIME_SESSION_PREPARED_INPUT}" \
  --emit-run-manifest "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}" \
  2>"${RUNTIME_SESSION_PREPARE_BINDING_STDERR}"; then
  echo "Error: runtime-session unknown input binding unexpectedly succeeded" >&2
  exit 1
fi
grep -q "references unknown external input binding: missing" \
  "${RUNTIME_SESSION_PREPARE_BINDING_STDERR}"
if build/bin/runtime-session \
  --artifact-manifest "${RUNTIME_SESSION_ARTIFACT_MANIFEST}" \
  --artifact-root "${FAKE_ARTIFACT_ROOT}" \
  --shape-arg arg0_dim0=128 \
  --output "out0=${RUNTIME_SESSION_PREPARED_OUTPUT}" \
  --emit-run-manifest "${RUNTIME_SESSION_PREPARED_RUN_MANIFEST}" \
  2>"${RUNTIME_SESSION_PREPARE_BINDING_STDERR}"; then
  echo "Error: runtime-session ambiguous output binding unexpectedly succeeded" >&2
  exit 1
fi
grep -q "binding is ambiguous; use task.binding selector: out0" \
  "${RUNTIME_SESSION_PREPARE_BINDING_STDERR}"

cat > "${RUNTIME_SESSION_CASE_JSON}" <<EOF
{
  "schema_version": 1,
  "artifact": {
    "root": "${FAKE_ARTIFACT_ROOT}",
    "manifest": "${RUNTIME_SESSION_ARTIFACT_MANIFEST}"
  },
  "backend": {
    "kind": "sim"
  },
  "shape_args": {
    "arg0": [128]
  },
  "inputs": [
    {
      "name": "kernel_a.arg0",
      "path": "${RUNTIME_SESSION_PREPARED_INPUT}"
    }
  ],
  "outputs": [
    {
      "name": "kernel_b.out0",
      "path": "${RUNTIME_SESSION_CASE_DIR}/custom_actual.npy"
    }
  ],
  "expected_outputs": [
    {
      "name": "kernel_b.out0",
      "path": "${RUNTIME_SESSION_PREPARED_EXPECTED}"
    }
  ],
  "validation": {
    "atol": 0.01,
    "rtol": 0.02
  }
}
EOF
build/bin/runtime-session \
  --case "${RUNTIME_SESSION_CASE_JSON}" \
  --emit-run-manifest "${RUNTIME_SESSION_CASE_RUN_MANIFEST}" \
  >"${RUNTIME_SESSION_CASE_DIR}/prepare.log"
grep -q "^run_manifest.path=${RUNTIME_SESSION_CASE_RUN_MANIFEST}$" \
  "${RUNTIME_SESSION_CASE_DIR}/prepare.log"
test -f "${RUNTIME_SESSION_CASE_RUN_MANIFEST}"
python3 - \
  "${RUNTIME_SESSION_CASE_RUN_MANIFEST}" \
  "${FAKE_ARTIFACT_ROOT}" \
  "${RUNTIME_SESSION_PREPARED_INPUT}" \
  "${RUNTIME_SESSION_PREPARED_EXPECTED}" \
  "${RUNTIME_SESSION_CASE_DIR}" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
artifact_root = sys.argv[2]
input_path = sys.argv[3]
expected_path = sys.argv[4]
case_dir = pathlib.Path(sys.argv[5])
tasks = manifest["tasks"]
assert manifest["backend"] == "sim"
assert manifest["artifact_root"] == artifact_root
assert [task["task_id"] for task in tasks] == ["kernel_a", "kernel_b"]
assert tasks[0]["workspace_size"] == 384
assert tasks[0]["block_dim"] == 4
assert tasks[0]["tiling"]["binary"].endswith(".tiling.bin")
assert tasks[0]["inputs"] == [{
    "name": "arg0",
    "path": input_path,
    "shape": [128],
    "dtype": "f16",
}]
assert tasks[0]["outputs"][0]["path"] == str(case_dir / "outputs" / "kernel_a.out0.actual.npy")
assert tasks[1]["outputs"][0]["path"] == str(case_dir / "custom_actual.npy")
assert tasks[1]["expected_outputs"] == [{
    "name": "out0",
    "path": expected_path,
    "shape": [128],
    "dtype": "f16",
}]
assert tasks[1]["atol"] == 0.01
assert tasks[1]["rtol"] == 0.02
PY
if build/bin/runtime-session \
  --case "${RUNTIME_SESSION_CASE_JSON}" \
  --artifact-manifest "${RUNTIME_SESSION_ARTIFACT_MANIFEST}" \
  --emit-run-manifest "${RUNTIME_SESSION_CASE_RUN_MANIFEST}" \
  2>"${RUNTIME_SESSION_CASE_CONFLICT_STDERR}"; then
  echo "Error: runtime-session --case conflict unexpectedly succeeded" >&2
  exit 1
fi
grep -q -- "--case cannot be combined" "${RUNTIME_SESSION_CASE_CONFLICT_STDERR}"

cat > "${RUNTIME_SESSION_CASE_DIR}/case_npu.json" <<EOF
{
  "schema_version": 1,
  "artifact": {
    "root": "${FAKE_ARTIFACT_ROOT}",
    "manifest": "${RUNTIME_SESSION_ARTIFACT_MANIFEST}"
  },
  "backend": {
    "kind": "npu"
  },
  "shape_args": {
    "arg0": [128]
  },
  "inputs": [
    {
      "name": "kernel_a.arg0",
      "path": "${RUNTIME_SESSION_PREPARED_INPUT}"
    }
  ]
}
EOF
build/bin/runtime-session \
  --case "${RUNTIME_SESSION_CASE_DIR}/case_npu.json" \
  --emit-run-manifest "${RUNTIME_SESSION_CASE_DIR}/case_npu_run_manifest.json" \
  --testing-driver npu-success \
  --run >"${RUNTIME_SESSION_CASE_DIR}/case_npu_run.log"
grep -q "^run_manifest.path=${RUNTIME_SESSION_CASE_DIR}/case_npu_run_manifest.json$" \
  "${RUNTIME_SESSION_CASE_DIR}/case_npu_run.log"
grep -q '^session.backend=npu$' "${RUNTIME_SESSION_CASE_DIR}/case_npu_run.log"
grep -q '^session.result=success$' "${RUNTIME_SESSION_CASE_DIR}/case_npu_run.log"
grep -q '^session.profile.count=2$' "${RUNTIME_SESSION_CASE_DIR}/case_npu_run.log"

echo "--- Checking runtime-session positive vec simulation path ---"
runtime_verify_build_example_toolchain
bash examples/relu-broadcast-transpose/run.sh >/tmp/runtime_session_example.log 2>&1
build/bin/runtime-session \
  --kernel examples/relu-broadcast-transpose/build_mainline/step10_kernel.cpp \
  --kernel-kind vec \
  --name relu_transpose_broadcast_add \
  --output "${RUNTIME_SESSION_ARTIFACT_ROOT}" \
  >/tmp/runtime_session_compile.log 2>&1
test -f "${RUNTIME_SESSION_ARTIFACT_ROOT}/out/manifest.txt"
build/bin/runtime-session \
  --kernel examples/relu-broadcast-transpose/build_mainline/step10_kernel.cpp \
  --kernel-kind vec \
  --name relu_transpose_broadcast_add \
  --output "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}" \
  >/tmp/runtime_session_compile_consumer.log 2>&1
test -f "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}/out/manifest.txt"

cat > "${RUNTIME_SESSION_RUN_MANIFEST}" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${RUNTIME_SESSION_ARTIFACT_ROOT}",
  "inputs": [
    { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data0.npy" },
    { "name": "data1", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data1.npy" }
  ],
  "outputs": [
    {
      "name": "out",
      "path": "${RUNTIME_SESSION_ACTUAL_OUTPUT}",
      "shape": [500, 640],
      "dtype": "f16"
    }
  ],
  "tiling": {
    "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/phase5_tiling_space.json",
    "params": "TB_M=32,TB_N=32,dim_arg0_0=640,dim_arg1_0=500,dim_arg1_1=640,dim_arg0_1=1"
  },
  "block_dim": 20,
  "workspace_size": 16777216,
  "profiling": true
}
EOF

build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_RUN_MANIFEST}" \
  --run >/tmp/runtime_session_run.log 2>&1
test -f "${RUNTIME_SESSION_ACTUAL_OUTPUT}"
grep -q '^session.profile.session_id=' /tmp/runtime_session_run.log
grep -q '^session.profile.count=' /tmp/runtime_session_run.log
grep -q '^session.profile.summary=' /tmp/runtime_session_run.log
grep -q '^session.runtime.attribute.scheduler_mode=serial$' /tmp/runtime_session_run.log
grep -q '^session.runtime.attribute.scheduler_stream_model=enabled$' /tmp/runtime_session_run.log
grep -q '^session.runtime.attribute.simulator_launch_model=dispatch_thread$' /tmp/runtime_session_run.log
grep -q '^session.runtime.counter.planned_task_count=1$' /tmp/runtime_session_run.log
grep -q '^session.runtime.counter.scheduler.stream.capacity_total=' /tmp/runtime_session_run.log
grep -q '^session.runtime.counter.serialized_launch_count=1$' /tmp/runtime_session_run.log
RUNTIME_SESSION_RUN_SUMMARY="$(sed -n 's/^session\.profile\.summary=//p' /tmp/runtime_session_run.log | head -n1)"
test -f "${RUNTIME_SESSION_RUN_SUMMARY}"
printf 'session.profile.summary=%s\n' "${RUNTIME_SESSION_RUN_SUMMARY}"

echo "--- Checking runtime-session DAG simulation path ---"
cat > "${RUNTIME_SESSION_DAG_MANIFEST}" <<EOF
{
  "backend": "sim",
  "artifact_root": "${RUNTIME_SESSION_ARTIFACT_ROOT}",
  "tasks": [
    {
      "task_id": "producer_a",
      "inputs": [
        { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data0.npy" },
        { "name": "data1", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data1.npy" }
      ],
      "outputs": [
        { "name": "mid_a", "shape": [500, 640], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/phase5_tiling_space.json",
        "params": "TB_M=32,TB_N=32,dim_arg0_0=640,dim_arg1_0=500,dim_arg1_1=640,dim_arg0_1=1"
      },
      "block_dim": 20,
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "producer_b",
      "inputs": [
        { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data0.npy" },
        { "name": "data1", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data1.npy" }
      ],
      "outputs": [
        { "name": "mid_b", "shape": [500, 640], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/phase5_tiling_space.json",
        "params": "TB_M=32,TB_N=32,dim_arg0_0=640,dim_arg1_0=500,dim_arg1_1=640,dim_arg0_1=1"
      },
      "block_dim": 20,
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "consumer",
      "dependencies": ["producer_a", "producer_b"],
      "artifact_root": "${RUNTIME_SESSION_SECOND_ARTIFACT_ROOT}",
      "inputs": [
        { "name": "data0", "path": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/input_data0.npy" },
        { "name": "data1", "source": "task_output", "upstream_task": "producer_a", "upstream_output": "mid_a" }
      ],
      "outputs": [
        { "name": "out", "path": "${RUNTIME_SESSION_DAG_OUTPUT}", "shape": [500, 640], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PROJECT_ROOT}/examples/relu-broadcast-transpose/build_mainline/phase5_tiling_space.json",
        "params": "TB_M=32,TB_N=32,dim_arg0_0=640,dim_arg1_0=500,dim_arg1_1=640,dim_arg0_1=1"
      },
      "block_dim": 20,
      "workspace_size": 16777216,
      "profiling": true
    }
  ]
}
EOF

PLAN_OUTPUT="$(build/bin/runtime-session --run-manifest "${RUNTIME_SESSION_DAG_MANIFEST}")"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[0\\]=producer_a"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[1\\]=producer_b"
printf '%s\n' "${PLAN_OUTPUT}" | grep -q "session.plan\\[2\\]=consumer"
build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_DAG_MANIFEST}" \
  --run >/tmp/runtime_session_dag_run.log 2>&1
test -f "${RUNTIME_SESSION_DAG_OUTPUT}"
grep -q '^session.profile.session_id=' /tmp/runtime_session_dag_run.log
grep -q '^session.profile.count=' /tmp/runtime_session_dag_run.log
grep -q '^session.profile.count=3$' /tmp/runtime_session_dag_run.log
grep -q '^session.profile.summary=' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_scope=global$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_mode=concurrent$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_policy=global_session_round_robin_baseline$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_fairness_policy=session_round_robin$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_priority_policy=static_session_priority$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_quota_policy=session_admission_quota$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_default_priority_class=normal$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.simulator_launch_model=dispatch_thread$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.planned_task_count=3$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.frontier_count=2$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.max_frontier_width=2$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.max_in_flight_tasks=2$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.quota.blocked=' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.policy.default_max_admitted_tasks=0$' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.fairness.session_order_size=' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.fairness.session_rotations_total=' /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.serialized_launch_count=3$' /tmp/runtime_session_dag_run.log
RUNTIME_SESSION_DAG_SUMMARY="$(sed -n 's/^session\.profile\.summary=//p' /tmp/runtime_session_dag_run.log | head -n1)"
test -f "${RUNTIME_SESSION_DAG_SUMMARY}"
printf 'session.profile.summary=%s\n' "${RUNTIME_SESSION_DAG_SUMMARY}"

echo "--- Checking runtime-session NPU path reaches unified backend ---"
cat > "${RUNTIME_SESSION_NPU_MANIFEST}" <<EOF
{
  "backend": "npu",
  "artifact_root": "${FAKE_ARTIFACT_ROOT}",
  "tasks": [
    {
      "task_id": "main",
      "outputs": [
        { "name": "out", "path": "${RUNTIME_SESSION_NPU_OUTPUT}", "shape": [4], "dtype": "f16" }
      ]
    }
  ]
}
EOF
NPU_STDERR="$(mktemp)"
if build/bin/runtime-session --run-manifest "${RUNTIME_SESSION_NPU_MANIFEST}" --run 2>"${NPU_STDERR}"; then
  echo "Error: runtime-session npu path unexpectedly succeeded" >&2
  exit 1
fi
grep -q '^session.backend=npu' "${NPU_STDERR}"
grep -q '^session.result=error' "${NPU_STDERR}"
grep -q '^session.error_stage=executor_initialize' "${NPU_STDERR}"

echo "--- Checking runtime-session NPU mock success path ---"
build/bin/runtime-session \
  --run-manifest "${RUNTIME_SESSION_NPU_MANIFEST}" \
  --testing-driver npu-success \
  --run >"${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}" 2>"${RUNTIME_SESSION_NPU_SUCCESS_STDERR}"
grep -q '^session.backend=npu' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.result=success' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile.session_id=' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile.count=1' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"
grep -q '^session.profile\[0\]=' "${RUNTIME_SESSION_NPU_SUCCESS_STDOUT}"

echo "--- Checking NPU smoke manifest helper assets ---"
bash test/tools/runtime/test_prepare_npu_smoke_manifests.sh
bash test/tools/runtime/test_prepare_real_npu_microcases.sh
bash test/tools/runtime/test_prepare_real_npu_multikernel.sh
bash test/tools/runtime/test_real_npu_transformer_case_registration.sh
bash test/tools/runtime/test_real_npu_skip_sim_registration.sh
bash test/tools/runtime/test_example_prepare_manifest_usage.sh
bash test/tools/runtime/test_broadcast_prepare_tiling.sh

# Compile test drivers
echo "--- Compiling runtime tests ---"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $(runtime_verify_cann_tiling_link_flags) \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_RUNTIME_BIN"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_taskgraph_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $(runtime_verify_cann_tiling_link_flags) \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_TASKGRAPH_RUNTIME_BIN"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I "$LLVM_SOURCE_INCLUDE" \
    test/tools/runtime/test_capi_runtime.cpp \
    build/lib/libAFIRRuntimeCAPI.so \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
    -ldl \
    -o "$TEST_CAPI_RUNTIME_BIN"

# Run
echo "--- Running test_taskgraph_runtime ---"
"$TEST_TASKGRAPH_RUNTIME_BIN"
echo "--- Running test_capi_runtime ---"
LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)" \
  "$TEST_CAPI_RUNTIME_BIN"
echo "--- Running test_runtime ---"
if "$TEST_RUNTIME_BIN"; then
  echo "--- Running SimBackend smoke baseline ---"
  export RUNTIME_VERIFY_RUNTIME_CORE_READY=1
  export RUNTIME_VERIFY_EXAMPLE_TOOLCHAIN_READY=1
  export RUNTIME_VERIFY_MIX_COMPILER_READY=1
  bash test/tools/runtime/run_simbackend_smoke.sh
  echo "--- Running repeated mix simulation baseline ---"
  bash test/tools/runtime/run_mix_repeat.sh
  exit 0
else
  STATUS=$?
  if [ "$STATUS" -eq 139 ]; then
    echo "Note: /tmp/test_runtime still segfaults in xvm after the task-graph runtime checks pass." >&2
  fi
  exit "$STATUS"
fi
