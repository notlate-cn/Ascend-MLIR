#!/usr/bin/env bash
# examples/matmul-add-leakyrelu/run.sh
# End-to-end pipeline: linalg IR -> AscendC kernel -> RuntimeMix mix validator
#
# Usage (on xvm):
#   source examples/env.sh
#   bash examples/matmul-add-leakyrelu/run.sh [--log]
#
# Dependencies: afir-opt, afir-translate, clang++, llvm-config, python3, numpy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/home/niu/code/llvm-project/llvm/build}"
BOOTSTRAP_BUILD_DIR="${BOOTSTRAP_BUILD_DIR:-${REPO_ROOT}/build/runtime-mix-bootstrap}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/build/runtime-mix-matmul-add-leakyrelu}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data}"

VERBOSE=false
for arg in "$@"; do [[ $arg == "--log" ]] && VERBOSE=true; done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " matmul + add(bias[N]) + leaky_relu AFIR pipeline"
echo "========================================================"

# ── STAGE 2: Transform Tiling ────────────────────────────────────────────────
echo ""
echo "=== [STAGE 2] Transform Tiling ==="
$AFIR_OPT --transform-interpreter \
  "$SCRIPT_DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step2_tiled.mlir"
log "  step2_tiled.mlir done"

# ── STAGE 3: Bufferize ───────────────────────────────────────────────────────
echo "=== [STAGE 3] Bufferize ==="
$AFIR_OPT \
  '--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map' \
  "$SCRIPT_DIR/step2_tiled.mlir" \
  --cse \
  -o "$SCRIPT_DIR/step3_bufferized.mlir"
log "  step3_bufferized.mlir done"

# ── STAGE 4: Buffer Placement ────────────────────────────────────────────────
echo "=== [STAGE 4] Buffer Placement ==="
$AFIR_OPT --ascendc-buffer-placement \
  "$SCRIPT_DIR/step3_bufferized.mlir" \
  -o "$SCRIPT_DIR/step4_buffer_placement.mlir"
log "  step4_buffer_placement.mlir done"

# ── STAGE 5: linalg -> AscendC ──────────────────────────────────────────────
echo "=== [STAGE 5] linalg-to-ascendc ==="
$AFIR_OPT --linalg-to-ascendc \
  "$SCRIPT_DIR/step4_buffer_placement.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step5_ascendc.mlir"
log "  step5_ascendc.mlir done"

# ── STAGE 6: Parallelize ─────────────────────────────────────────────────────
echo "=== [STAGE 6] Parallelize ==="
$AFIR_OPT --ascendc-parallelize \
  "$SCRIPT_DIR/step5_ascendc.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step6_parallelize.mlir"
log "  step6_parallelize.mlir done"

# ── STAGE 7: Prepare for emit ────────────────────────────────────────────────
echo "=== [STAGE 7] Prepare for emit ==="
$AFIR_OPT --ascendc-prepare-for-emit \
  "$SCRIPT_DIR/step6_parallelize.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step7_kernel.mlir"
$AFIR_OPT --canonicalize-cann-signature \
  "$SCRIPT_DIR/step7_kernel.mlir" \
  -o "$SCRIPT_DIR/step7_cann.mlir"
log "  step7_kernel.mlir, step7_cann.mlir done"

# ── STAGE 8: Codegen ─────────────────────────────────────────────────────────
echo "=== [STAGE 8] Codegen ==="
$AFIR_TRANSLATE -mlir-to-cann \
  "$SCRIPT_DIR/step7_cann.mlir" \
  -o "$SCRIPT_DIR/step8_kernel.cpp"
log "  step8_kernel.cpp done"

# ── Bootstrap mix-compiler + mix-validator ───────────────────────────────────
echo ""
echo "=== Bootstrap RuntimeMix tools ==="

if [[ ! -d "${LLVM_BUILD_DIR}" ]]; then
  echo "LLVM build dir not found: ${LLVM_BUILD_DIR}" >&2
  exit 2
fi

mkdir -p "${BOOTSTRAP_BUILD_DIR}/bin"
LLVM_FLAGS="$("${LLVM_BUILD_DIR}/bin/llvm-config" --cxxflags --ldflags --libs support --system-libs)"

if [[ ! -x "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/tools/mix-compiler/mix_compiler_main.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixDirectBackend.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixAbi.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/NpyIO.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixAbiExtractor.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixStubTemplate.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixSourceAnalyzer.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]]; then
  clang++ \
    "${REPO_ROOT}/tools/mix-compiler/mix_compiler_main.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixDirectBackend.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixAbi.cpp" \
    "${REPO_ROOT}/lib/Runtime/NpyIO.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixAbiExtractor.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixCommandBuilder.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixSourceAnalyzer.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixStubTemplate.cpp" \
    ${LLVM_FLAGS} \
    -std=c++17 \
    -I"${REPO_ROOT}/include" \
    -o "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler"
fi

if [[ ! -x "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" ]] || \
   [[ "${REPO_ROOT}/tools/mix-validator/mix_validator_main.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/Executor.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" ]] || \
   [[ "${REPO_ROOT}/lib/Runtime/MixAbi.cpp" -nt "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" ]]; then
  clang++ \
    "${REPO_ROOT}/tools/mix-validator/mix_validator_main.cpp" \
    "${REPO_ROOT}/lib/Runtime/Executor.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixAbi.cpp" \
    ${LLVM_FLAGS} \
    -std=c++17 \
    -I"${REPO_ROOT}/include" \
    -o "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator"
fi

# ── Stage 8b: Materialize official-style mix kernel ───────────────────────────
# Feed RuntimeMix the original __global__ kernel entry and let toolkit
# preprocess/extract_host_stub generate the auto_gen wrapper and launcher.
echo "=== [STAGE 8b] Generate official-style mix kernel ==="
python3 - "$SCRIPT_DIR/fc_leakyrelu_official_style.cpp" <<'PYEOF'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
out_path.write_text("""#define __FC_LEAKYRELU_WRAPPERLESS_KERNEL_FUN_H__

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint64_t *dst = reinterpret_cast<uint64_t *>(tiling);
    auto tiling64 = reinterpret_cast<__gm__ uint64_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i) {
        dst[i] = tiling64[i];
    }
}

extern "C" __global__ __aicore__ void fc_leakyrelu(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    (void)workspace;

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
               MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

        GlobalTensor<half> aGM, bGM;
        GlobalTensor<float> cGM, biasGM;
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), tiling.M * tiling.Ka);
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), tiling.Kb * tiling.N);
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), tiling.M * tiling.N);
        biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), tiling.N);

        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        mm.SetTensorA(aGM);
        mm.SetTensorB(bGM);
        mm.SetBias(biasGM);
        mm.template IterateAll(cGM);
        mm.End();
        CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }

    if ASCEND_IS_AIV {
        TQue<TPosition::VECIN, 1> reluInQueue;
        TQue<TPosition::VECOUT, 1> reluOutQueue;

        uint32_t count = (uint32_t)(tiling.singleCoreM * tiling.singleCoreN / 2);
        GlobalTensor<float> cGM;
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + GetBlockIdx() * count);

        pipe.InitBuffer(reluInQueue, 1, count * sizeof(float));
        pipe.InitBuffer(reluOutQueue, 1, count * sizeof(float));

        CrossCoreWaitFlag(3);

        LocalTensor<float> reluInLocal = reluInQueue.AllocTensor<float>();
        DataCopy(reluInLocal, cGM, count);
        reluInQueue.EnQue<float>(reluInLocal);

        LocalTensor<float> inLocal = reluInQueue.DeQue<float>();
        LocalTensor<float> outLocal = reluOutQueue.AllocTensor<float>();
        LeakyRelu(outLocal, inLocal, (float)0.001f, count);
        reluOutQueue.EnQue<float>(outLocal);
        reluInQueue.FreeTensor(inLocal);

        LocalTensor<float> finalLocal = reluOutQueue.DeQue<float>();
        DataCopy(cGM, finalLocal, count);
        reluOutQueue.FreeTensor(finalLocal);
    }
}
""")
print("  fc_leakyrelu_official_style.cpp written")
PYEOF

rm -rf "${ARTIFACT_DIR}"

# ── Generate test data ────────────────────────────────────────────────────────
echo "=== [STAGE 9] Generate test data ==="
mkdir -p "${DATA_DIR}/input" "${DATA_DIR}/output" "${DATA_DIR}/npy"
python3 "$SCRIPT_DIR/gen_data.py" \
  --M 128 --K 256 --N 128 --seed 42 \
  --out-dir "${DATA_DIR}/npy"

# ── RuntimeMix compile ────────────────────────────────────────────────────────
echo "=== [STAGE 10] RuntimeMix compile ==="
"${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" \
  --kernel "$SCRIPT_DIR/fc_leakyrelu_official_style.cpp" \
  --cann-mlir "$SCRIPT_DIR/step7_cann.mlir" \
  --npy-dir "${DATA_DIR}/npy" \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}"

read -r ACTUAL_OUTPUT_PATH GOLDEN_OUTPUT_PATH < <(python3 - "${DATA_DIR}" "${ARTIFACT_DIR}/out/manifest.txt" <<'PY'
import sys
from pathlib import Path
import numpy as np

data_dir = Path(sys.argv[1])
npy_dir = data_dir / "npy"
inp_dir = data_dir / "input"
out_dir = data_dir / "output"
manifest_path = Path(sys.argv[2])

def read_manifest(path):
    manifest = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key] = value
    return manifest

manifest = read_manifest(manifest_path)
inp_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
kernel_name = (manifest.get("abi_runtime_kernel_name") or
               manifest.get("abi_logical_kernel_name") or
               manifest.get("kernel_name") or
               manifest.get("requested_kernel_name") or "")

input_count = int(manifest["abi_input_count"])
for idx in range(input_count):
    name = manifest[f"abi_input{idx}_name"]
    runtime_file = manifest.get(f"abi_input{idx}_file") or \
        manifest.get(f"abi_input{idx}_runtime_file")
    if not runtime_file and kernel_name:
        runtime_file = f"{kernel_name}.{name}.input.bin"
    if not runtime_file:
        raise SystemExit(f"missing runtime file for input {idx}")
    npy_path = npy_dir / f"{name}.npy"
    if not npy_path.exists():
        npy_path = npy_dir / f"input{idx}.npy"
    np.load(npy_path).tofile(inp_dir / runtime_file)

output_count = int(manifest["abi_output_count"])
if output_count != 1:
    raise SystemExit(f"expected one output, got {output_count}")
output_name = manifest["abi_output0_name"]
runtime_output = manifest.get("abi_output0_file") or \
    manifest.get("abi_output0_runtime_file")
golden_output = manifest.get("abi_output0_golden_file")
if not runtime_output and kernel_name:
    runtime_output = f"{kernel_name}.{output_name}.output.bin"
if not golden_output and kernel_name:
    golden_output = f"{kernel_name}.{output_name}.golden.bin"
if not runtime_output:
    raise SystemExit("missing runtime output file in manifest")
if not golden_output:
    raise SystemExit("missing golden output file in manifest")
output_npy = npy_dir / f"{output_name}.npy"
if not output_npy.exists():
    output_npy = npy_dir / "output0.npy"
np.load(output_npy).tofile(out_dir / golden_output)
print((manifest_path.parent.parent / runtime_output).resolve(), (out_dir / golden_output).resolve())
PY
)

# ── mix-validator simulation ──────────────────────────────────────────────────
echo "=== [STAGE 11] mix-validator ==="
[[ -f "${ARTIFACT_DIR}/out/tiling.bin" ]]

"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${DATA_DIR}/input" \
  --golden "${GOLDEN_OUTPUT_PATH}" \
  --soc "${SOC_VERSION}"

python3 - "${GOLDEN_OUTPUT_PATH}" "${ACTUAL_OUTPUT_PATH}" <<'PY'
import sys
import numpy as np

golden = np.fromfile(sys.argv[1], dtype=np.float32)
actual = np.fromfile(sys.argv[2], dtype=np.float32)
if golden.shape != actual.shape:
    raise SystemExit(f"shape mismatch: {actual.shape} vs {golden.shape}")
diff = np.abs(actual - golden)
print(f"max_abs_diff={diff.max():.6e}")
print(f"mean_abs_diff={diff.mean():.6e}")
if not np.allclose(actual, golden, atol=1.0, rtol=1e-2):
    raise SystemExit("FAIL: outputs differ beyond tolerance")
print("PASS")
PY

echo "artifact_dir=${ARTIFACT_DIR}"
echo "data_dir=${DATA_DIR}"
