#!/usr/bin/env bash
# Probe 3 CV-fusion variants beyond matmul+relu to map out gaps:
#   A. matmul + unary negate            (other unary elementwise epilogue)
#   B. matmul + bias-add  (no relu)     (pure binary epilogue with bias)
#   C. matmul + bias-bcast(1xN→MxN)     (binary epilogue with explicit broadcast)
#
# For each: walk the pipeline stage-by-stage and report which stage rejects.
# Does NOT attempt sim (these may not get past mlir-to-cann's relu-only
# epilogue detector).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"

W="${W:-/tmp/cv-fusion-probes}"
rm -rf "${W}"; mkdir -p "${W}"

# Common M/K/N
M=64; K=32; N=64

probe() {
  local name="$1"
  local funcname="${name//-/_}"
  local mlir="${W}/${name}/step0.mlir"
  local cann="${W}/${name}/step7_cann.mlir"
  local cpp="${W}/${name}/step8.cpp"
  mkdir -p "${W}/${name}"
  cat > "${mlir}" <<< "$2"

  local stage1 stage2 epilogue
  if ${AFIR_OPT} --vector-plan-codegen "${mlir}" -o "${cann}" 2> "${W}/${name}/opt.log"; then
    stage1=PASS
    # Pull the recorded epilogue kind for context.
    epilogue="$(grep -oE 'abi_matmul_epilogue_kind = "[^"]+"' "${cann}" | head -1 || true)"
    if ${AFIR_TRANSLATE} --mlir-to-cann "${cann}" -o "${cpp}" 2> "${W}/${name}/translate.log"; then
      stage2=PASS
    else
      stage2="FAIL: $(head -1 "${W}/${name}/translate.log" | sed 's|.*error: ||' | cut -c1-90)"
    fi
  else
    stage1="FAIL: $(head -1 "${W}/${name}/opt.log" | sed 's|.*error: ||' | cut -c1-90)"
    stage2="(skipped)"
    epilogue="(n/a)"
  fi
  printf "%-26s  codegen=%-30s  emit=%s\n  ↳ %s\n" \
         "${name}" "${stage1}" "${stage2}" "${epilogue:-(no attr)}"
}

# ── Case A: matmul + negf (unary, not relu) ─────────────────────────────────
probe "A-mm-negate" "$(cat <<MLIR
func.func @A_mm_negate(%a: tensor<${M}x${K}xf16>, %b: tensor<${K}x${N}xf16>, %init: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>) outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %e = tensor.empty() : tensor<${M}x${N}xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%c : tensor<${M}x${N}xf32>) outs(%e : tensor<${M}x${N}xf32>) {
  ^bb0(%v: f32, %_: f32):
    %t = arith.negf %v : f32
    linalg.yield %t : f32
  } -> tensor<${M}x${N}xf32>
  return %r : tensor<${M}x${N}xf32>
}
MLIR
)"

# ── Case B: matmul + bias-add (no relu) ──────────────────────────────────────
# bias is rank-1 [N], broadcast across rows via indexing_maps.
probe "B-mm-bias-add" "$(cat <<MLIR
func.func @B_mm_bias_add(%a: tensor<${M}x${K}xf16>, %b: tensor<${K}x${N}xf16>, %init: tensor<${M}x${N}xf32>, %bias: tensor<${N}xf32>) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>) outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %e = tensor.empty() : tensor<${M}x${N}xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%c, %bias : tensor<${M}x${N}xf32>, tensor<${N}xf32>) outs(%e : tensor<${M}x${N}xf32>) {
  ^bb0(%v: f32, %bv: f32, %_: f32):
    %t = arith.addf %v, %bv : f32
    linalg.yield %t : f32
  } -> tensor<${M}x${N}xf32>
  return %r : tensor<${M}x${N}xf32>
}
MLIR
)"

# ── Case C: matmul + bias-bcast(1xN→MxN) + relu ──────────────────────────────
# bias is rank-2 [1, N], requires explicit broadcast through indexing_maps.
probe "C-mm-bcast-bias-relu" "$(cat <<MLIR
func.func @C_mm_bcast_bias_relu(%a: tensor<${M}x${K}xf16>, %b: tensor<${K}x${N}xf16>, %init: tensor<${M}x${N}xf32>, %bias: tensor<1x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %c = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>) outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %e = tensor.empty() : tensor<${M}x${N}xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%c, %bias : tensor<${M}x${N}xf32>, tensor<1x${N}xf32>) outs(%e : tensor<${M}x${N}xf32>) {
  ^bb0(%v: f32, %bv: f32, %_: f32):
    %s = arith.addf %v, %bv : f32
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %s, %z : f32
    linalg.yield %t : f32
  } -> tensor<${M}x${N}xf32>
  return %r : tensor<${M}x${N}xf32>
}
MLIR
)"

echo "Artifacts under ${W}; see step0.mlir / step7_cann.mlir / opt.log / translate.log per case."
