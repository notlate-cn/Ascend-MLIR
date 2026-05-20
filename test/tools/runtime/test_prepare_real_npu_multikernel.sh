#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HELPER="${PROJECT_ROOT}/examples/real-npu-multikernel/prepare.sh"

TMP_DIR="$(mktemp -d /tmp/runtime-real-npu-multikernel-test.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

bash "${HELPER}" --out-dir "${TMP_DIR}/case" --skip-compile

python3 - <<PY
import ast
import json
import struct
from pathlib import Path

root = Path("${TMP_DIR}/case")


def load_manifest(case_name):
    path = root / case_name / "run_manifest.json"
    assert path.exists(), path
    return json.loads(path.read_text())


def find_task(manifest, task_id):
    for task in manifest["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise AssertionError(f"missing task: {task_id}")


def load_f16_npy(path):
    data = path.read_bytes()
    assert data[:6] == b"\\x93NUMPY", path
    assert data[6:8] == b"\\x01\\x00", path
    header_len = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_len].decode("latin1"))
    values = [
        struct.unpack("<e", data[i:i + 2])[0]
        for i in range(10 + header_len, len(data), 2)
    ]
    return header["shape"], values


serial = load_manifest("serial-two-kernel")
assert serial["backend"] == "npu"
assert [task["task_id"] for task in serial["tasks"]] == [
    "producer",
    "consumer",
]
serial_producer = find_task(serial, "producer")
serial_consumer = find_task(serial, "consumer")
assert Path(serial_consumer["artifact_root"]).parent.name == "copy_scalar640"
assert serial_producer["outputs"][0]["name"] == "mid"
assert serial_producer["outputs"][0]["shape"] == [640]
assert serial_producer["outputs"][0]["dtype"] == "f16"
assert serial_consumer["dependencies"] == ["producer"]
assert serial_consumer["inputs"][0]["source"] == "task_output"
assert serial_consumer["inputs"][0]["upstream_task"] == "producer"
assert serial_consumer["inputs"][0]["upstream_output"] == "mid"
assert serial_consumer["outputs"][0]["shape"] == [640]
assert serial_consumer["expected_outputs"][0]["name"] == "out"

fork = load_manifest("fork-join")
assert fork["backend"] == "npu"
assert [task["task_id"] for task in fork["tasks"]] == [
    "producer_a",
    "producer_b",
    "consumer",
]
consumer = find_task(fork, "consumer")
assert Path(consumer["artifact_root"]).parent.name == "add_wait640"
assert consumer["dependencies"] == ["producer_a", "producer_b"]
assert consumer["inputs"][0]["source"] == "task_output"
assert consumer["inputs"][0]["upstream_task"] == "producer_a"
assert consumer["inputs"][0]["upstream_output"] == "a"
assert consumer["inputs"][1]["source"] == "task_output"
assert consumer["inputs"][1]["upstream_task"] == "producer_b"
assert consumer["inputs"][1]["upstream_output"] == "b"
assert consumer["outputs"][0]["shape"] == [640]
assert consumer["expected_outputs"][0]["name"] == "out"

for kernel_name in [
    "const640",
    "copy_scalar640",
    "const_with_input640",
    "add_wait640",
]:
    source = root / "artifacts" / kernel_name / f"{kernel_name}.cpp"
    assert source.exists(), source

shape, serial_expected = load_f16_npy(
    root / "serial-two-kernel" / "expected.npy"
)
assert shape == (640,)
assert serial_expected == [1.0] * 640

shape, fork_expected = load_f16_npy(root / "fork-join" / "expected.npy")
assert shape == (640,)
assert fork_expected == [3.0] * 640
PY

echo "real npu multikernel helper test passed"
