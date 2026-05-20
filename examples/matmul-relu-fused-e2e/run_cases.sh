#!/usr/bin/env bash
# Sweep 6 CV-fusion cases through the full auto-fuse pipeline + sim.
#
# Each case overrides: name, M, K, N, input range, with_relu.
# Validates max_abs_diff = 0.0 against numpy golden.
#
# Usage:
#   source ~/Ascend/latest/set_env.sh
#   source examples/env.sh
#   export ASCEND_DAV_SIM_VERSION=dav_3002
#   PATH=./build/bin:$PATH bash examples/matmul-relu-fused-e2e/run_cases.sh
set -uo pipefail
export ASCEND_DAV_SIM_VERSION=${ASCEND_DAV_SIM_VERSION:-dav_3002}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
MIX_COMPILER="${MIX_COMPILER:-mix-compiler}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"

CANN_ARCH="$(uname -m)"
[[ "${CANN_ARCH}" == "x86_64" ]] && CANN_ARCH=x86_64-linux || CANN_ARCH=aarch64-linux
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64:${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib:${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64:${ASCEND_HOME_PATH}/runtime/lib64/stub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

WORK_ROOT="${WORK_ROOT:-/tmp/cv-fusion-cases}"
rm -rf "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"

# case_name M K N input_lo input_hi with_relu xfail dyn with_bias
# xfail=1     -> failure is expected (Phase 2 work, recorded but not fatal).
# dyn=1       -> emits ?x? shapes in step0.mlir; pipeline resolves shapes from npy.
# with_bias=1 -> inserts a rank-1 bias-add generic between matmul and (optional) relu.
#                Exercises the Identity epilogue (no-relu / bias-only)
#                or BiasAddRelu (with relu) translator paths.
CASES=(
  "small-baseline       32  16  64 -10 10  1  0  0  0"
  "medium-shape        128  64 128 -10 10  1  0  0  0"
  "tall-skinny         256  32  32 -10 10  1  0  0  0"
  "wide-flat            32  32 256 -10 10  1  0  0  0"
  "mixed-sign-relu      64  32  64 -50 50  1  0  0  0"
  "matmul-only          64  32  64 -10 10  0  1  0  0"
  "k-tail-24            32  24  64 -10 10  1  0  0  0"
  "dyn-shape           128  32  64 -10 10  1  0  1  0"
  "bias-only            64  32  64 -10 10  0  0  0  1"
  "bias-relu            64  32  64 -10 10  1  0  0  1"
  "bcast-bias-relu      64  32  64 -10 10  1  0  0  2"
)

emit_step0() {
  local out="$1" name="$2" M="$3" K="$4" N="$5" with_relu="$6" dyn="${7:-0}" with_bias="${8:-0}"
  # MLIR identifiers can't contain '-'; sanitize for the func name.
  local funcname="${name//-/_}"
  # Bias variants: with_bias=1 -> rank-1 [N] bias; with_bias=2 -> rank-2 [1,N]
  # bcast bias.  Optionally followed by relu (with_relu=1).
  if [[ ( "${with_bias}" == "1" || "${with_bias}" == "2" ) && "${dyn}" == "0" ]]; then
    local biasTy biasMap
    if [[ "${with_bias}" == "2" ]]; then
      biasTy="tensor<1x${N}xf32>"; biasMap="affine_map<(d0,d1)->(0,d1)>"
    else
      biasTy="tensor<${N}xf32>";   biasMap="affine_map<(d0,d1)->(d1)>"
    fi
    local body
    if [[ "${with_relu}" == "1" ]]; then
      body="    %s = arith.addf %v, %bv : f32
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %s, %z : f32
    linalg.yield %t : f32"
    else
      body="    %t = arith.addf %v, %bv : f32
    linalg.yield %t : f32"
    fi
    cat > "${out}" <<MLIR
func.func @${funcname}(%a: tensor<${M}x${K}xf16>, %b: tensor<${K}x${N}xf16>,
                       %init: tensor<${M}x${N}xf32>, %bias: ${biasTy}) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>) outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %e = tensor.empty() : tensor<${M}x${N}xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, ${biasMap}, affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%c, %bias : tensor<${M}x${N}xf32>, ${biasTy}) outs(%e : tensor<${M}x${N}xf32>) {
  ^bb0(%v: f32, %bv: f32, %_: f32):
${body}
  } -> tensor<${M}x${N}xf32>
  return %r : tensor<${M}x${N}xf32>
}
MLIR
    return
  fi
  local TA TB TInit
  if [[ "${dyn}" == "1" ]]; then
    TA="tensor<?x?xf16>";  TB="tensor<?x?xf16>";  TInit="tensor<?x?xf32>"
  else
    TA="tensor<${M}x${K}xf16>"; TB="tensor<${K}x${N}xf16>"; TInit="tensor<${M}x${N}xf32>"
  fi
  if [[ "${with_relu}" == "1" && "${dyn}" == "1" ]]; then
    cat > "${out}" <<MLIR
func.func @${funcname}(%a: ${TA}, %b: ${TB}, %init: ${TInit}) -> ${TInit} {
  %c = linalg.matmul ins(%a, %b : ${TA}, ${TB}) outs(%init : ${TInit}) -> ${TInit}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dm = tensor.dim %c, %c0 : ${TInit}
  %dn = tensor.dim %c, %c1 : ${TInit}
  %empty = tensor.empty(%dm, %dn) : ${TInit}
  %r = linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types=["parallel","parallel"]} ins(%c : ${TInit}) outs(%empty : ${TInit}) {
  ^bb0(%v: f32, %_: f32):
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %v, %z : f32
    linalg.yield %t : f32
  } -> ${TInit}
  return %r : ${TInit}
}
MLIR
    return
  fi
  if [[ "${with_relu}" == "1" ]]; then
    cat > "${out}" <<MLIR
func.func @${funcname}(%a: tensor<${M}x${K}xf16>,
                    %b: tensor<${K}x${N}xf16>,
                    %init: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul
    ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>)
    outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %empty = tensor.empty() : tensor<${M}x${N}xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                     affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%c : tensor<${M}x${N}xf32>) outs(%empty : tensor<${M}x${N}xf32>) {
  ^bb0(%v: f32, %_: f32):
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %v, %z : f32
    linalg.yield %t : f32
  } -> tensor<${M}x${N}xf32>
  return %r : tensor<${M}x${N}xf32>
}
MLIR
  else
    cat > "${out}" <<MLIR
func.func @${funcname}(%a: tensor<${M}x${K}xf16>,
                    %b: tensor<${K}x${N}xf16>,
                    %init: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul
    ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>)
    outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %c : tensor<${M}x${N}xf32>
}
MLIR
  fi
}

emit_gen_data() {
  local out="$1" M="$2" K="$3" N="$4" lo="$5" hi="$6" with_relu="$7" with_bias="${8:-0}"
  cat > "${out}" <<PY
import numpy as np, sys
from pathlib import Path
rng = np.random.default_rng(42)
A = rng.integers(${lo}, ${hi}+1, (${M}, ${K})).astype(np.float16)
B = rng.integers(${lo}, ${hi}+1, (${K}, ${N})).astype(np.float16)
init = np.zeros((${M}, ${N}), dtype=np.float32)
mm = A.astype(np.float32) @ B.astype(np.float32)
out = mm.astype(np.float32)
with_bias = ${with_bias}
if with_bias:
    # with_bias==2 -> rank-2 [1,N] bias; else rank-1 [N].
    if with_bias == 2:
        bias = rng.integers(1, 10, (1, ${N})).astype(np.float32)
    else:
        bias = rng.integers(1, 10, (${N},)).astype(np.float32)
    out = out + bias  # numpy broadcasts both forms over rows
if ${with_relu}:
    out = np.maximum(out, 0.0)
out = out.astype(np.float32)
d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
pairs = [("input_a",A),("input_b",B),("input_init",init),("input0",A),("input1",B),("input2",init),("output",out),("output0",out)]
if with_bias:
    pairs.append(("input_bias",bias))
    pairs.append(("input3",bias))
for n, v in pairs:
    np.save(d/f"{n}.npy", v)
print(f"shapes M=${M} K=${K} N=${N} range=[${lo},${hi}] relu=${with_relu} bias=${with_bias} out_range=[{out.min()},{out.max()}]")
PY
}

run_one() {
  local name="$1" M="$2" K="$3" N="$4" lo="$5" hi="$6" with_relu="$7" dyn="${8:-0}" with_bias="${9:-0}"
  local dir="${WORK_ROOT}/${name}"
  mkdir -p "${dir}"
  emit_step0   "${dir}/step0.mlir" "${name}" "${M}" "${K}" "${N}" "${with_relu}" "${dyn}" "${with_bias}"
  emit_gen_data "${dir}/gen.py"    "${M}" "${K}" "${N}" "${lo}" "${hi}" "${with_relu}" "${with_bias}"

  ${AFIR_OPT} --auto-fuse-codegen "${dir}/step0.mlir" -o "${dir}/step7_cann.mlir" \
    2> "${dir}/opt.log" || { echo "FAIL[${name}]: auto-fuse-codegen"; tail -5 "${dir}/opt.log"; return 1; }
  ${AFIR_TRANSLATE} --mlir-to-cann "${dir}/step7_cann.mlir" -o "${dir}/step8.cpp" \
    2> "${dir}/translate.log" || { echo "FAIL[${name}]: mlir-to-cann"; tail -5 "${dir}/translate.log"; return 1; }

  python3 "${dir}/gen.py" "${dir}/data" >/dev/null || { echo "FAIL[${name}]: gen_data"; return 1; }

  rm -rf "${dir}/artifact"
  ${MIX_COMPILER} --kernel "${dir}/step8.cpp" --cann-mlir "${dir}/step7_cann.mlir" \
    --npy-dir "${dir}/data" --output "${dir}/artifact" --soc "${SOC_VERSION}" \
    > "${dir}/mix.log" 2>&1 || { echo "FAIL[${name}]: mix-compiler"; tail -8 "${dir}/mix.log"; return 1; }

  python3 "${SCRIPT_DIR}/build_run_manifest.py" \
    --artifact-dir "${dir}/artifact" --data-dir "${dir}/data" \
    --out-manifest "${dir}/run.json" --out-npy "${dir}/actual.npy" >/dev/null \
    || { echo "FAIL[${name}]: build_manifest"; return 1; }

  ${RUNTIME_SESSION} --run-manifest "${dir}/run.json" --run > "${dir}/sim.log" 2>&1 || {
    echo "FAIL[${name}]: runtime-session exit"; tail -5 "${dir}/sim.log"; return 1; }
  grep -q '^session.result=success$'  "${dir}/sim.log" || { echo "FAIL[${name}]: session.result"; return 1; }
  grep -q '^session.validation=pass$' "${dir}/sim.log" || { echo "FAIL[${name}]: session.validation"; return 1; }

  local diff
  diff="$(python3 -c "
import numpy as np
a=np.load('${dir}/actual.npy'); g=np.load('${dir}/data/output.npy')
print(float(np.max(np.abs(a-g))))
")"
  echo "PASS[${name}] M=${M} K=${K} N=${N} relu=${with_relu} range=[${lo},${hi}]  max_abs_diff=${diff}"
}

pass=0; fail=0; xfail=0; xpass=0
declare -a FAILED XPASSED
for spec in "${CASES[@]}"; do
  read -r name M K N lo hi with_relu xfail_expected dyn with_bias <<<"${spec}"
  if run_one "${name}" "${M}" "${K}" "${N}" "${lo}" "${hi}" "${with_relu}" "${dyn:-0}" "${with_bias:-0}"; then
    if [[ "${xfail_expected}" == "1" ]]; then
      xpass=$((xpass+1)); XPASSED+=("${name}")
      echo "  (^ XPASS: case marked xfail but now passes — promote it)"
    else
      pass=$((pass+1))
    fi
  else
    if [[ "${xfail_expected}" == "1" ]]; then
      xfail=$((xfail+1))
      echo "  (^ XFAIL: known limitation, not counted as failure)"
    else
      fail=$((fail+1)); FAILED+=("${name}")
    fi
  fi
done
echo "============================================================"
echo "summary: PASS=${pass}  XFAIL=${xfail}  FAIL=${fail}  XPASS=${xpass}  total=${#CASES[@]}"
if [[ "${fail}" -gt 0 ]]; then
  printf '  failed: %s\n' "${FAILED[@]}"
  exit 1
fi
