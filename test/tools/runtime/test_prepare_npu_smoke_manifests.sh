#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/prepare_npu_smoke_manifests.sh"

TMP_DIR="$(mktemp -d /tmp/runtime-npu-smoke-test.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

VEC_MANIFEST="${TMP_DIR}/vec_manifest.json"
MIX_MANIFEST="${TMP_DIR}/mix_manifest.json"

bash "${HELPER}" vec \
  "${VEC_MANIFEST}" \
  "/tmp/vec-artifact" \
  "/tmp/input_data0.npy" \
  "/tmp/input_data1.npy" \
  "/tmp/vec_output.npy" \
  "/tmp/vec_expected.npy"

grep -q '"backend": "npu"' "${VEC_MANIFEST}"
grep -q '"artifact_root": "/tmp/vec-artifact"' "${VEC_MANIFEST}"
grep -q '"name": "data0"' "${VEC_MANIFEST}"
grep -q '"path": "/tmp/input_data0.npy"' "${VEC_MANIFEST}"
grep -q '"name": "data1"' "${VEC_MANIFEST}"
grep -q '"path": "/tmp/input_data1.npy"' "${VEC_MANIFEST}"
grep -q '"path": "/tmp/vec_output.npy"' "${VEC_MANIFEST}"
grep -q '"path": "/tmp/vec_expected.npy"' "${VEC_MANIFEST}"

bash "${HELPER}" mix \
  "${MIX_MANIFEST}" \
  "/tmp/mix-artifact" \
  "/tmp/input_a.npy" \
  "/tmp/input_b.npy" \
  "/tmp/input_bias.npy" \
  "/tmp/mix_output.npy" \
  "/tmp/mix_expected.npy"

grep -q '"backend": "npu"' "${MIX_MANIFEST}"
grep -q '"artifact_root": "/tmp/mix-artifact"' "${MIX_MANIFEST}"
grep -q '"name": "lhs"' "${MIX_MANIFEST}"
grep -q '"path": "/tmp/input_a.npy"' "${MIX_MANIFEST}"
grep -q '"name": "rhs"' "${MIX_MANIFEST}"
grep -q '"path": "/tmp/input_b.npy"' "${MIX_MANIFEST}"
grep -q '"name": "bias"' "${MIX_MANIFEST}"
grep -q '"path": "/tmp/input_bias.npy"' "${MIX_MANIFEST}"
grep -q '"path": "/tmp/mix_output.npy"' "${MIX_MANIFEST}"
grep -q '"path": "/tmp/mix_expected.npy"' "${MIX_MANIFEST}"

echo "npu smoke manifest helper test passed"
