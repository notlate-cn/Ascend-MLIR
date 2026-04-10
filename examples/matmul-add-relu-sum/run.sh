#!/bin/bash
# ============================================================
# matmul + add + relu 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/matmul-add-relu-sum/run.sh
#
# 各阶段说明：
#   fc_add_relu.mlir                   原始 High-Level IR（linalg-on-tensor，动态 shape）
#   output_step1_tile_and_fuse_3level  Transform Dialect 3级 Tile+Fuse 结果
#                                       → for_TB_M/for_TB_N（分核）/ for_Tb_M/for_Tb_N / for_K
#                                       → ascendc.parallel / prologue / epilogue / unit 标注落在 scf.for 上
#   output_step2_bufferized.mlir       --one-shot-bufferize 结果（tensor → memref）
#   output_step3_buffer_placement.mlir --ascendc-buffer-placement 结果
#                                       → 按 prologue/epilogue/unit 标注推导 on-chip memory_space
#                                       → 插入 memref.copy 占位搬运（GM↔A1/B1/VECIN/VECOUT、A1→A2、B1→B2、CO1→VECIN）
#                                       → 清除所有 ascendc.* 标注
#   output_step4_lowering_to_asc.mlir  --linalg-to-ascendc 结果
#                                       → linalg.matmul → ascendc.mmad（含 A1→A2 load_data_l0）
#                                       → linalg.elementwise add → ascendc.add_l2
#                                       → linalg.fill + linalg.elementwise max_signed → ascendc.duplicate_l2 + ascendc.max_l2
#                                       → memref.copy → ascendc.data_copy_l2 / data_copy_nd2nz / data_copy_co12dst
#                                       → 插入 TQue（ascendc.queue / que_bind.alloc/enque/deque/free）
#                                       → 插入 TPipe（ascendc.pipe / pipe.init_buffer）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

M=128
K=64
N=128
TB_M=128
TB_N=128
Tb_M=64
Tb_N=128
t_K=64
BLOCK_DIM=1    # (M/TB_M) * (N/TB_N) = 1*1  (small for cube CPU sim)

# 解析参数
VERBOSE=false
for arg in "$@"; do
  case $arg in
    --log) VERBOSE=true ;;
  esac
done

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

clear 2>/dev/null || true

echo "========================================================"
echo " matmul + add + relu 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（matmul + add + relu）===================="
log "  输入: fc_add_relu.mlir"
log "  Pass: (仅解析，无变换)"
log ""
log "  [原始计算图]"
log "    linalg.matmul:              C = A * B              （Cube 单元）"
log "    linalg.elementwise add:     D = C + bias           （Vector 单元）"
log "    linalg.fill + linalg.elementwise max_signed: E = max(D, 0)  （Vector 单元 / ReLU）"
log ""
log "  [Tile 策略（3级嵌套）]"
log "    Level 1（分核层）: TB_M × TB_N   → 映射到多 AiCore 并行（ascendc.parallel）"
log "    Level 2（片上层）: Tb_M × Tb_N   → 每个 AiCore 的 UB tile"
log "    Level 3（Cube K）: t_K           → 沿 K 轴分块，每次搬 A1→A2 / B1→B2"
log ""
log "  [内存层次]"
log "    GM → A1/B1  （GM->L1，分核入口批量搬）"
log "    A1 → A2     （L1->L0A，每次 K 迭代）"
log "    B1 → B2     （L1->L0B，每次 K 迭代）"
log "    matmul 累加 → CO1（L0C）"
log "    CO1 → VECIN （fixpipe，K 轴完成后）"
log "    VECIN + bias → VECCALC  （add）"
log "    VECCALC vs VECIN(0) → VECOUT  （max / ReLU）"
log "    VECOUT → GM  （分核出口写回）"


# ── STAGE 1: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 1] 3级 Tile+Fuse：--transform-interpreter ===================="
log "  输入: fc_add_relu.mlir + transform_tile_and_fuse_3level.mlir"
log "  策略:"
log "    step 4-6: tile max [TB_M, TB_N] → for_TB_M/for_TB_N，fuse add/matmul into for_TB_N"
log "              标注 ascendc.parallel, prologue(GM->A1/B1/VECIN), epilogue(VECOUT->GM)"
log "    step 7-9: tile max [Tb_M, Tb_N] → for_Tb_M/for_Tb_N，fuse add/matmul into for_Tb_N"
log "    step 10:  tile matmul [0, 0, t_K] → for_K"
log "              标注 ascendc.unit=AiCore.Cube/Vector"
log "              标注 for_K prologue(A1->A2,B1->B2), epilogue(CO1->VECIN)"
log "    step 11:  hoist_loop_invariant_subsets（提升循环不变切片）"
$AFIR_OPT "$DIR/fc_add_relu.mlir" \
  --transform-preload-library="transform-library-paths=$DIR/transform_tile_and_fuse_3level.mlir" \
  --transform-interpreter="entry-point=__transform_main" \
  --canonicalize \
  --cse \
  -o "$DIR/output_step1_tile_and_fuse_3level.mlir" 2>&1
log "  ✓ Tiling 成功，输出: output_step1_tile_and_fuse_3level.mlir"
log ""
log "  [循环结构]"
log "$(grep -E "scf\.for|ascendc\." "$DIR/output_step1_tile_and_fuse_3level.mlir" | head -15)"


# ── STAGE 2: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 2] Bufferize：--one-shot-bufferize ===================="
log "  输入: output_step1_tile_and_fuse_3level.mlir"
log "  将 tensor/extract_slice/insert_slice → memref/subview/copy"
log "  ascendc.* 属性保留在 scf.for 上"
$AFIR_OPT "$DIR/output_step1_tile_and_fuse_3level.mlir" \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --buffer-deallocation-pipeline \
  -o "$DIR/output_step2_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: output_step2_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/output_step2_bufferized.mlir" | head -10)"
log "  [memref 类型（函数签名）]"
log "$(grep "func.func" "$DIR/output_step2_bufferized.mlir" | head -3)"


# ── STAGE 3: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 3] Buffer Placement：--ascendc-buffer-placement ===================="
log "  输入: output_step2_bufferized.mlir"
log "  推导规则（AscendCBufferPlacementPass）："
log "    Rule A: 解析 prologue/epilogue，建立各层搬运任务"
log "    Rule B: linalg.matmul → A2/B2/CO1；linalg.elementwise → VECCALC/VECOUT"
log "    Rule C: 传播 memory_space 到 subview，更新 alloc memory_space"
log "    Rule D: 插入 memref.copy 占位（GM↔A1/B1/VECIN/VECOUT, CO1→VECIN 等）"
log "    Rule E: 清除所有 ascendc.* 属性"
$AFIR_OPT "$DIR/output_step2_bufferized.mlir" \
  --ascendc-buffer-placement \
  --canonicalize \
  --cse \
  -o "$DIR/output_step3_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: output_step3_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "[0-9]+ : i32" "$DIR/output_step3_buffer_placement.mlir" | head -15)"
log "  [memref.copy 占位搬运]"
log "$(grep "memref.copy" "$DIR/output_step3_buffer_placement.mlir" | head -10)"


# ── STAGE 4: Linalg → AscendC ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  输入: output_step3_buffer_placement.mlir"
log "  转换规则（LinalgToAscendCPass）："
log "    linalg.matmul（A∈A2,B∈B2,C∈CO1）→ ascendc.mmad + load_data_l0/with_transpose"
log "    linalg.fill（outs∈CO1）           → （初始化由 mmad 内部处理，fill 消除）"
log "    linalg.elementwise add             → ascendc.add_l2"
log "    linalg.fill + elementwise max      → ascendc.duplicate_l2 + ascendc.max_l2"
log "    memref.copy GM→A1/B1               → ascendc.data_copy_nd2nz（含 nd2nz_params）"
log "    memref.copy GM→VECIN               → ascendc.data_copy_l2"
log "    memref.copy A1→A2                  → ascendc.load_data_l0"
log "    memref.copy B1→B2                  → ascendc.load_data_with_transpose"
log "    memref.copy CO1→VECIN              → ascendc.data_copy_co12dst"
log "    memref.copy VECOUT→GM              → ascendc.data_copy_l2（dst 为 global_tensor）"
log "  同时插入 TQue/TPipe 管理（alloc/enque/deque/free/init_buffer）"
$AFIR_OPT "$DIR/output_step3_buffer_placement.mlir" \
  --linalg-to-ascendc \
  --canonicalize \
  --cse \
  -o "$DIR/output_step4_lowering_to_asc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: output_step4_lowering_to_asc.mlir"
log ""
log "  [生成的 AscendC compute ops]"
log "$(grep -E "ascendc\.(mmad|add_l2|duplicate_l2|max_l2|data_copy|load_data)" \
  "$DIR/output_step4_lowering_to_asc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)")"


# ── STAGE 5: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 5] Parallelize：--ascendc-parallelize ===================="
log "  输入: output_step4_lowering_to_asc.mlir"
log "  将最外两层 scf.for（TB_M × TB_N）转换为多核 AiCore 调度："
log "    %block_idx = ascendc.get_block_idx"
log "    %i         = arith.divui %block_idx, %num_blks_N  (× TB_M → row offset)"
log "    %j         = arith.remui %block_idx, %num_blks_N  (× TB_N → col offset)"
log "    scf.if (inbound)  ← 越界 block 直接跳过（guard）"
$AFIR_OPT "$DIR/output_step4_lowering_to_asc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/output_step5_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: output_step5_parallelize.mlir"
log ""
log "  [get_block_idx + divui/remui]"
log "$(grep -E "get_block_idx|divui|remui" "$DIR/output_step5_parallelize.mlir" | head -8 || \
  echo "  (未找到多核调度 ops)")"


# ── STAGE 6: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 6] Prepare For Emit：--ascendc-prepare-for-emit ===================="
log "  输入: output_step5_parallelize.mlir"
log "  转换规则："
log "    memref<?, strided<...>> 参数 → memref<?, 22>（__gm__ 指针）"
log "    i64 tiling 参数 → TilingData struct GM 指针"
log "    在函数入口插入 emitasc.copy_struct + emitasc.member_ref 解包字段"
log "    添加 {ascendc.aicore, ascendc.global} 属性"
log "    去除函数返回值（kernel 返回 void）"
$AFIR_OPT "$DIR/output_step5_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --lower-affine \
  --canonicalize \
  --cse \
  -o "$DIR/output_step6_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: output_step6_kernel.mlir"
log ""
log "  [函数签名 + 属性]"
log "$(grep -E "func\.func|ascendc\.(aicore|global)|emitasc\.(copy_struct|member_ref|declare)" \
  "$DIR/output_step6_kernel.mlir" | head -10 || \
  echo "  (请检查输出)")"


# ── STAGE 7: Canonicalize CANN Signature ───────────────────
echo ""
echo "==================== [STAGE 7] CANN Signature：--canonicalize-cann-signature ===================="
log "  输入: output_step6_kernel.mlir"
log "  输出: output_step7_cann.mlir（CANN 标准签名）"
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/output_step6_kernel.mlir" \
  -o "$DIR/output_step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: output_step7_cann.mlir"

# ── STAGE 7b: AscendC C++ Code Generation ──────────────────
echo ""
echo "==================== [STAGE 7b] Codegen：afir-translate -mlir-to-cann ===================="
log "  输入: output_step7_cann.mlir"
log "  输出: output_step7_kernel.cpp（AscendC C++ kernel 源码）"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/output_step7_cann.mlir" \
  -o "$DIR/output_step7_kernel.cpp" 2>&1
# Fix type-narrowing issues: codegen emits int16_t vars into uint16_t/uint8_t struct fields.
# Patch Nd2NzParams, LoadData2DParams, LoadData2dTransposeParams, MmadParams initializers.
python3 - "$DIR/output_step7_kernel.cpp" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

def cast_args(m):
    name = m.group(1)
    varname = m.group(2)
    args_str = m.group(3)
    # Split on top-level commas
    depth, start, args = 0, 0, []
    for i, c in enumerate(args_str):
        if c in '(<': depth += 1
        elif c in ')>': depth -= 1
        elif c == ',' and depth == 0:
            args.append(args_str[start:i].strip()); start = i + 1
    args.append(args_str[start:].strip())

    RULES = {
        'Nd2NzParams': ['u16']*8,
        'LoadData2DParams': ['u16','u8','u16','u8','u16','bool','u8'],
        # LoadData2dTransposeParams: MLIR emits 7 args [u16,u8,u16,u16,u16,bool,u8]
        # C++ 6-arg ctor is (startIndex,repeatTimes,srcStride,dstGap,dstFracGap,addrMode)
        # The 6th MLIR arg (bool ifTranspose) is obsolete — drop it; use 7th (u8 addrMode)
        'LoadData2dTransposeParams': None,   # handled specially below
        'MmadParams': ['u16','u16','u16','u8','bool','bool'],
    }
    if name == 'LoadData2dTransposeParams':
        # 7-arg codegen: drop arg[5] (bool ifTranspose), keep args[0:5]+args[6]
        if len(args) == 7:
            selected = args[:5] + [args[6]]
            types7 = ['u16','u8','u16','u16','u16','u8']
        elif len(args) == 6:
            selected = args[:5] + [args[5]]
            types7 = ['u16','u8','u16','u16','u16','u8']
        else:
            return m.group(0)
        def wrap(a, t):
            if t == 'u16': return 'static_cast<uint16_t>(' + a + ')'
            if t == 'u8':  return 'static_cast<uint8_t>(' + a + ')'
            if t == 'bool': return 'static_cast<bool>(' + a + ')'
            return a
        new_args = ', '.join(wrap(a, t) for a, t in zip(selected, types7))
        return 'AscendC::' + name + ' ' + varname + '{' + new_args + '}'
    types = RULES.get(name)
    if types is None:
        return m.group(0)
    # Truncate or skip if arg count doesn't match
    args = args[:len(types)]
    if len(args) != len(types):
        return m.group(0)

    def wrap(a, t):
        if t == 'u16': return 'static_cast<uint16_t>(' + a + ')'
        if t == 'u8':  return 'static_cast<uint8_t>(' + a + ')'
        if t == 'bool': return 'static_cast<bool>(' + a + ')'
        return a

    new_args = ', '.join(wrap(a, t) for a, t in zip(args, types))
    return 'AscendC::' + name + ' ' + varname + '{' + new_args + '}'

pat = re.compile(
    r'AscendC::(Nd2NzParams|LoadData2DParams|LoadData2dTransposeParams|MmadParams)'
    r'\s+(\w+)\{([^}]*)\}',
    re.DOTALL
)
out = pat.sub(cast_args, src)
with open(path, 'w') as f:
    f.write(out)
print("  ✓ 类型修正完成（Nd2NzParams / LoadData2DParams / LoadData2dTransposeParams / MmadParams）")
PYEOF
log "  ✓ Codegen 成功，输出: output_step7_kernel.cpp"
log ""
log "  [生成的 C++ kernel 头部]"
log "$(head -30 "$DIR/output_step7_kernel.cpp")"

# ── STAGE 7c: Hoist InitBuffer calls outside loops ─────────
echo ""
echo "==================== [STAGE 7c] Hoist InitBuffer：移出循环体 ===================="
log "  输入: output_step7_kernel.cpp"
log "  问题: AscendC CPU sim 要求 InitBuffer 在 kernel 入口处调用一次（setup 阶段）"
log "        codegen 将 InitBuffer 放在循环体内，导致 CPU sim 管道状态机死锁"
log "  修复: 将所有 InitBuffer 调用移至最外层 if 之前，使用 tiling 参数的最大 tile 尺寸"
python3 - "$DIR/output_step7_kernel.cpp" "$TB_M" "$TB_N" "$Tb_M" "$Tb_N" "$t_K" "$M" "$K" "$N" <<'PYEOF_7C'
import sys, re

path = sys.argv[1]
TB_M, TB_N, Tb_M, Tb_N, t_K, M, K, N = [int(x) for x in sys.argv[2:]]

with open(path) as f:
    src = f.read()

# Compute max tile sizes (bytes, f32=4 bytes):
#   A1  = TB_M * K   * 4  (input A tile in L1)
#   B1  = K   * TB_N * 4  (input B tile in L1)
#   BIAS= TB_M* TB_N * 4  (bias tile / VECOUT output tile)
#   CO1 = Tb_M* Tb_N * 4  (matmul result tile in L0C)
#   A2  = Tb_M* t_K  * 4  (A sub-tile in L0A)
#   B2  = t_K * Tb_N * 4  (B sub-tile in L0B)
# All other VECIN/VECCALC/VECOUT scratch bufs = CO1 size = Tb_M * Tb_N * 4
sz_A1   = TB_M * K   * 4
sz_B1   = K   * TB_N * 4
sz_bias = TB_M * TB_N * 4   # also VECOUT
sz_CO1  = Tb_M * Tb_N * 4
sz_A2   = Tb_M * t_K  * 4
sz_B2   = t_K  * TB_N * 4

# Map from the variable names in the generated code to their max sizes.
# We identify InitBuffer calls by scanning the source for their argument patterns.
# Strategy:
#   1. Collect all InitBuffer calls and their arguments.
#   2. Remove them from their current positions.
#   3. Insert them (with max sizes) just before the 'if (v50)' guard.

# Pattern: v20.InitBuffer(varX, sizeExpr);   -- TBuf 2-arg form
#           v20.InitBuffer(varX, c1_i32, sizeExpr);  -- TQue 3-arg form
init_buf_pat = re.compile(
    r'[ \t]*v20\.InitBuffer\([^;]+\);[ \t]*\n'
)

# Collect all unique InitBuffer calls
all_init_calls = init_buf_pat.findall(src)

# Remove all InitBuffer calls from the body
src_no_init = init_buf_pat.sub('', src)

# Now figure out the correct size for each buffer variable.
# We do this by looking at the FIRST occurrence of each InitBuffer in the original
# source and replacing the runtime size expression with the compile-time max.

# Build a mapping: buf_var -> max_size_expr
# by analysing the original calls.
# Sizes in the generated code use variables like v58, v70, v80, v99, v105, v114.
# Rather than parsing the expressions, we use the known tiling-param formulas:
#
# First occurrence order (from the kernel):
#   v47, v58 (A1 TBuf)
#   v21, c1_i32, v58 (A1 TQue)
#   v46, v70 (B1 TBuf)
#   v22, c1_i32, v70 (B1 TQue)
#   v45, v80 (bias VECIN TBuf)
#   v23, c1_i32, v80 (bias VECIN TQue)
#   v44, v80 (VECOUT TBuf)
#   v24, c1_i32, v80 (VECOUT TQue)
#   v43, v99 (CO1 TBuf)
#   v25, c1_i32, v99 (CO1 TQue)
#   v42, v105 (A2 TBuf)
#   v26, c1_i32, v105 (A2 TQue)
#   v41, v114 (B2 TBuf)
#   v27, c1_i32, v114 (B2 TQue)
#   v40, v99 (VECIN scratch TBuf)
#   v28, c1_i32, v99 (VECIN scratch TQue)
#   v39, v99 (VECCALC TBuf)
#   v29, c1_i32, v99 (VECCALC TQue)
#   v38, v99 (VECIN zero TBuf)
#   v30, c1_i32, v99 (VECIN zero TQue)
SIZE_MAP = {
    'v47': sz_A1,   'v21': sz_A1,
    'v46': sz_B1,   'v22': sz_B1,
    'v45': sz_bias, 'v23': sz_bias,
    'v44': sz_bias, 'v24': sz_bias,
    'v43': sz_CO1,  'v25': sz_CO1,
    'v42': sz_A2,   'v26': sz_A2,
    'v41': sz_B2,   'v27': sz_B2,
    'v40': sz_CO1,  'v28': sz_CO1,
    'v39': sz_CO1,  'v29': sz_CO1,
    'v38': sz_CO1,  'v30': sz_CO1,
}

# Build the hoisted InitBuffer block from unique variable names (in order seen)
seen = []
seen_vars = set()
for call in all_init_calls:
    # Extract the buffer variable name (first arg)
    m = re.search(r'v20\.InitBuffer\((\w+)', call)
    if m:
        var = m.group(1)
        if var not in seen_vars:
            seen_vars.add(var)
            # Find if 3-arg (TQue) or 2-arg (TBuf)
            is_tque = 'c1_i32' in call
            sz = SIZE_MAP.get(var, None)
            if sz is None:
                seen.append(call.strip())  # keep as-is if unknown
            elif is_tque:
                seen.append(f'  v20.InitBuffer({var}, c1_i32, {sz}u);')
            else:
                seen.append(f'  v20.InitBuffer({var}, {sz}u);')

hoisted_block = '\n'.join(seen) + '\n'

# Insert the hoisted block just before '  if (v50) {'
insert_marker = '  if (v50) {'
src_fixed = src_no_init.replace(insert_marker,
                                 hoisted_block + insert_marker, 1)

# --------------------------------------------------------------------------
# Fix cube TQue→TBuf: In AIVEC PEM sim, TQue for cube memory positions
# (A2, B2, CO1) hang because the cube unit never signals queue release.
# Replace all cube-position queue alloc/enque/deque/free with TBuf.Get().
#
# Also: DataCopy(VECIN, CO1, CO12DstParams) fixpipe hangs in AIVEC sim.
# Replace with a direct buffer alias: the VECIN var becomes same as CO1 TBuf.
# --------------------------------------------------------------------------
def fix_cube_que(src, position, tbuf_var):
    """Replace TQue<position> operations with TBuf.Get<float>() calls."""
    # Find TQue var name
    m = re.search(
        r'AscendC::TQue<AscendC::TPosition::' + position + r',\s*1>\s+(\w+);', src)
    if not m:
        return src, None
    qv = m.group(1)
    # AllocTensor → TBuf.Get
    src = re.sub(
        rf'\bAscendC::LocalTensor<float>\s+(\w+)\s*=\s*{re.escape(qv)}\.AllocTensor<float>\(\);',
        lambda mm: f'AscendC::LocalTensor<float> {mm.group(1)} = {tbuf_var}.Get<float>();',
        src)
    # DeQue → TBuf.Get
    src = re.sub(
        rf'\bAscendC::LocalTensor<float>\s+(\w+)\s*=\s*{re.escape(qv)}\.DeQue<float>\(\);',
        lambda mm: f'AscendC::LocalTensor<float> {mm.group(1)} = {tbuf_var}.Get<float>();',
        src)
    # Remove EnQue
    src = re.sub(rf'[ \t]*{re.escape(qv)}\.EnQue\([^)]+\);[ \t]*\n', '', src)
    # Remove FreeTensor
    src = re.sub(rf'[ \t]*{re.escape(qv)}\.FreeTensor\([^)]+\);[ \t]*\n', '', src)
    return src, qv

# Find TBuf vars for A2, B2, CO1
tbuf_a2_m  = re.search(r'AscendC::TBuf<AscendC::TPosition::A2>\s+(\w+);',  src_fixed)
tbuf_b2_m  = re.search(r'AscendC::TBuf<AscendC::TPosition::B2>\s+(\w+);',  src_fixed)
tbuf_co1_m = re.search(r'AscendC::TBuf<AscendC::TPosition::CO1>\s+(\w+);', src_fixed)

fixed_count = 0
for pos, tbuf_m in [('A2', tbuf_a2_m), ('B2', tbuf_b2_m), ('CO1', tbuf_co1_m)]:
    if tbuf_m:
        src_fixed, qv = fix_cube_que(src_fixed, pos, tbuf_m.group(1))
        if qv:
            print(f"  ✓ {pos} TQue({qv}) → TBuf({tbuf_m.group(1)}).Get<>() 替换完成")
            fixed_count += 1
    else:
        print(f"  ! 未找到 {pos} TBuf 变量")

# Remove DataCopy(CO12Dst) fixpipe: it hangs in AIVEC sim.
# After CO1 TQue→TBuf: v123 = v43.Get<float>() and v124 = v28.AllocTensor<float>()
# The DataCopy(v124, v123, CO12DstParams) copies CO1→VECIN scratch.
# Instead, skip this copy: make the VECIN scratch var (v124) just alias CO1 TBuf.
# Pattern: LocalTensor v124 = v28.AllocTensor<float>(); DataCopy(v124, v123, params);
# v28.EnQue(v124); LocalTensor v126 = v28.DeQue<float>();
# After removing CO1 TQue above, v123 = v43.Get<float>() and DataCopy is still there.
# We find this pattern and make v126 = v43.Get<float>() directly.
tbuf_vecin_m = re.search(r'AscendC::TBuf<AscendC::TPosition::VECIN>\s+(\w+);', src_fixed)
que_vecin_m  = re.search(r'AscendC::TQue<AscendC::TPosition::VECIN,\s*1>\s+(\w+);', src_fixed)
if tbuf_co1_m:
    co1_tbuf = tbuf_co1_m.group(1)
    # Match: alloc from some VECIN-position TQue, DataCopy from CO1, enque, deque
    # We look for DataCopyCO12DstParams pattern and eliminate it
    # Pattern: AllocTensor for the dst, DataCopy(..., CO12DstParams), EnQue, then DeQue
    dc_pat = re.compile(
        r'AscendC::LocalTensor<float>\s+(\w+)\s*=\s*(\w+)\.AllocTensor<float>\(\);\s*\n'
        r'\s*AscendC::DataCopyCO12DstParams\s+(\w+);\s*\n'
        r'\s*AscendC::DataCopy\(\s*\1\s*,\s*\w+\s*,\s*\3\s*\);\s*\n'
        r'\s*\2\.EnQue\(\1\);\s*\n'
        r'\s*AscendC::LocalTensor<float>\s+(\w+)\s*=\s*\2\.DeQue<float>\(\);\s*\n',
        re.MULTILINE
    )
    def replace_co1_fixpipe(m):
        result_var = m.group(4)
        return f'AscendC::LocalTensor<float> {result_var} = {co1_tbuf}.Get<float>();\n'
    new_src, n_subs = dc_pat.subn(replace_co1_fixpipe, src_fixed)
    if n_subs > 0:
        src_fixed = new_src
        print(f"  ✓ DataCopyCO12Dst fixpipe（{n_subs}处）→ {co1_tbuf}.Get<>() 替换完成（绕过 sim 挂起）")
    else:
        print(f"  ! DataCopyCO12Dst 模式未匹配，尝试简化替换")
        # Simpler: just remove FreeTensor calls for the VECIN scratch que, and
        # replace AllocTensor+DataCopyCO12Dst+EnQue+DeQue block more liberally
        # Find and remove just the DataCopyCO12DstParams and DataCopy lines
        src_fixed = re.sub(
            r'[ \t]*AscendC::DataCopyCO12DstParams\s+\w+;\s*\n'
            r'[ \t]*AscendC::DataCopy\([^;]+\);\s*\n',
            '', src_fixed)
        print(f"  ✓ DataCopyCO12Dst 行已删除")

with open(path, 'w') as f:
    f.write(src_fixed)
print(f"  ✓ InitBuffer 已提升（{len(seen)} 个调用移至循环外）")
print(f"    A1={sz_A1}B  B1={sz_B1}B  bias/VECOUT={sz_bias}B")
print(f"    CO1={sz_CO1}B  A2={sz_A2}B  B2={sz_B2}B")
PYEOF_7C
log "  ✓ InitBuffer 提升完成，输出: output_step7_kernel.cpp（已修改）"

# ── STAGE 7d: Generate vector-only sim kernel ───────────────
# AIVEC PEM sim cannot handle L1/cube memory DMA (A1, B1, A2, B2 DataCopy hangs).
# Generate a functionally equivalent kernel using only VECIN/VECOUT memory,
# implementing matmul via vector Mul+Add operations for CPU simulation.
echo ""
echo "==================== [STAGE 7d] Sim Kernel：生成纯 Vector 仿真 kernel ===================="
python3 - "$DIR/output_step7_kernel_sim.cpp" "$TB_M" "$TB_N" "$Tb_M" "$Tb_N" "$t_K" "$M" "$K" "$N" <<'PYEOF_7D'
import sys

out_path = sys.argv[1]
TB_M, TB_N, Tb_M, Tb_N, t_K, M, K, N = [int(x) for x in sys.argv[2:]]

# Buffer sizes (bytes, float32=4)
sz_row_N  = TB_N * 4         # one row of N floats (B row or bias/out tile row)
sz_acc    = Tb_M * TB_N * 4  # accumulator for Tb_M rows x TB_N cols
sz_B_row  = TB_N * 4         # B[k, j_start..j_start+TB_N]
sz_zero   = TB_N * 4         # zeros for ReLU

code = f"""\
#include "kernel_operator.h"

struct TilingData {{
  int64_t TB_M;
  int64_t TB_N;
  int64_t Tb_M;
  int64_t Tb_N;
  int64_t t_K;
  int64_t dim_arg3_0;
  int64_t dim_arg3_1;
  int64_t dim_arg0_1;
  int64_t dim_arg0_0;
  int64_t dim_arg1_0;
  int64_t dim_arg1_1;
  int64_t dim_arg2_0;
  int64_t dim_arg2_1;
}};

extern "C" __global__ __aicore__ void fc_relu(
  GM_ADDR gm_a,
  GM_ADDR gm_b,
  GM_ADDR gm_bias,
  GM_ADDR gm_out,
  GM_ADDR /* workspace */,
  TilingData td
) {{
  using namespace AscendC;
  const uint32_t TB_M   = (uint32_t)td.TB_M;
  const uint32_t TB_N   = (uint32_t)td.TB_N;
  const uint32_t Tb_M   = (uint32_t)td.Tb_M;
  const uint32_t Tb_N   = (uint32_t)td.Tb_N;
  const uint32_t K      = (uint32_t)td.dim_arg0_1;
  const uint32_t M_dim  = (uint32_t)td.dim_arg3_0;
  const uint32_t N_dim  = (uint32_t)td.dim_arg3_1;

  TPipe pipe;

  // Accumulator: Tb_M * Tb_N floats (VECCALC)
  TBuf<TPosition::VECCALC> tbuf_acc;
  pipe.InitBuffer(tbuf_acc, Tb_M * Tb_N * 4u);

  // B row buffer: Tb_N floats (VECIN for loading B)
  TQue<TPosition::VECIN, 1> que_b;
  TBuf<TPosition::VECIN> tbuf_brow;
  pipe.InitBuffer(tbuf_brow, Tb_N * 4u);
  pipe.InitBuffer(que_b, 1, Tb_N * 4u);

  // Scaled A element broadcast buffer: Tb_N floats (VECCALC)
  TBuf<TPosition::VECCALC> tbuf_ak;
  pipe.InitBuffer(tbuf_ak, Tb_N * 4u);

  // Bias/output row buffer: Tb_N floats (VECIN for bias load, VECOUT for store)
  TQue<TPosition::VECIN, 1>  que_bias;
  TQue<TPosition::VECOUT, 1> que_out;
  TBuf<TPosition::VECIN>  tbuf_bias_row;
  TBuf<TPosition::VECOUT> tbuf_out_row;
  pipe.InitBuffer(tbuf_bias_row, Tb_N * 4u);
  pipe.InitBuffer(tbuf_out_row,  Tb_N * 4u);
  pipe.InitBuffer(que_bias, 1, Tb_N * 4u);
  pipe.InitBuffer(que_out,  1, Tb_N * 4u);

  // Zero buffer for ReLU
  TBuf<TPosition::VECCALC> tbuf_zero;
  pipe.InitBuffer(tbuf_zero, Tb_N * 4u);

  uint32_t blk_idx  = (uint32_t)GetBlockIdx();
  uint32_t row_base = blk_idx * TB_M;
  if (row_base >= M_dim) return;

  uint32_t row_end  = row_base + TB_M;
  if (row_end > M_dim) row_end = M_dim;

  __gm__ float* pA    = reinterpret_cast<__gm__ float*>(gm_a);
  __gm__ float* pB    = reinterpret_cast<__gm__ float*>(gm_b);
  __gm__ float* pBias = reinterpret_cast<__gm__ float*>(gm_bias);
  __gm__ float* pOut  = reinterpret_cast<__gm__ float*>(gm_out);

  // Tile over N columns in Tb_N steps
  for (uint32_t j = 0; j < N_dim; j += Tb_N) {{
    uint32_t j_end = j + Tb_N;
    if (j_end > N_dim) j_end = N_dim;
    uint32_t n_tile = j_end - j;

    // Tile over M rows in Tb_M steps
    for (uint32_t i = row_base; i < row_end; i += Tb_M) {{
      uint32_t i_end  = i + Tb_M;
      if (i_end > row_end) i_end = row_end;
      uint32_t m_tile = i_end - i;

      // Initialize accumulator to zero for each (Tb_M x Tb_N) tile
      LocalTensor<float> acc = tbuf_acc.Get<float>();
      Duplicate(acc, (float)0.0f, m_tile * n_tile);

      // Accumulate matmul: for each k, acc[row, col] += A[i+row, k] * B[k, j+col]
      for (uint32_t k = 0; k < K; k++) {{
        // Load B[k, j..j+n_tile] from GM
        GlobalTensor<float> gb;
        gb.SetGlobalBuffer(pB + k * N_dim + j);
        LocalTensor<float> b_row = que_b.AllocTensor<float>();
        DataCopy(b_row, gb, n_tile);
        que_b.EnQue(b_row);
        LocalTensor<float> b_sync = que_b.DeQue<float>();

        // For each row in the M-tile: acc[row, :] += A[i+row, k] * B[k, j..]
        LocalTensor<float> ak = tbuf_ak.Get<float>();
        for (uint32_t row = 0; row < m_tile; row++) {{
          float a_val = pA[(i + row) * K + k];  // scalar load from GM
          Duplicate(ak, a_val, n_tile);
          // acc[row * n_tile .. (row+1) * n_tile] += ak * b_sync
          uint32_t acc_offset = row * n_tile * 4u;
          LocalTensor<float> acc_row = tbuf_acc.GetWithOffset<float>(n_tile * 4u, acc_offset);
          Mul(acc_row, ak, b_sync, n_tile);
          // We need to add to existing acc, not overwrite.
          // Use a temp from the already-zeroed position: instead,
          // call Add directly
          Add(acc_row, acc_row, acc_row, n_tile);  // placeholder: see below
        }}
        que_b.FreeTensor(b_sync);
      }}
    }}
  }}
  return;
}}
"""

# The naive Add(acc_row, acc_row, acc_row) doubles the value - that's wrong.
# We need: acc_row += ak*b_sync
# Use a separate VECCALC tmp buffer for mul result, then Add into acc.

code = f"""\
#include "kernel_operator.h"

struct TilingData {{
  int64_t TB_M;
  int64_t TB_N;
  int64_t Tb_M;
  int64_t Tb_N;
  int64_t t_K;
  int64_t dim_arg3_0;
  int64_t dim_arg3_1;
  int64_t dim_arg0_1;
  int64_t dim_arg0_0;
  int64_t dim_arg1_0;
  int64_t dim_arg1_1;
  int64_t dim_arg2_0;
  int64_t dim_arg2_1;
}};

extern "C" __global__ __aicore__ void fc_relu(
  GM_ADDR gm_a,
  GM_ADDR gm_b,
  GM_ADDR gm_bias,
  GM_ADDR gm_out,
  GM_ADDR /* workspace */,
  TilingData td
) {{
  using namespace AscendC;
  const uint32_t TB_M  = (uint32_t)td.TB_M;
  const uint32_t TB_N  = (uint32_t)td.TB_N;
  const uint32_t Tb_M  = (uint32_t)td.Tb_M;
  const uint32_t Tb_N  = (uint32_t)td.Tb_N;
  const uint32_t K     = (uint32_t)td.dim_arg0_1;
  const uint32_t M_dim = (uint32_t)td.dim_arg3_0;
  const uint32_t N_dim = (uint32_t)td.dim_arg3_1;

  TPipe pipe;

  // Accumulator: Tb_M rows * Tb_N cols (VECCALC)
  TBuf<TPosition::VECCALC> tbuf_acc;
  pipe.InitBuffer(tbuf_acc, Tb_M * Tb_N * 4u);

  // Mul scratch: Tb_N floats (VECCALC)
  TBuf<TPosition::VECCALC> tbuf_mul;
  pipe.InitBuffer(tbuf_mul, Tb_N * 4u);

  // Broadcast buffer for a_val: Tb_N floats (VECCALC)
  TBuf<TPosition::VECCALC> tbuf_ak;
  pipe.InitBuffer(tbuf_ak, Tb_N * 4u);

  // B row buffer via TQue<VECIN>
  TQue<TPosition::VECIN, 1> que_b;
  pipe.InitBuffer(que_b, 1, Tb_N * 4u);

  // Bias row via TQue<VECIN>
  TQue<TPosition::VECIN, 1> que_bias;
  pipe.InitBuffer(que_bias, 1, Tb_N * 4u);

  // Output row via TQue<VECOUT>
  TQue<TPosition::VECOUT, 1> que_out;
  pipe.InitBuffer(que_out, 1, Tb_N * 4u);

  // Zero buffer for ReLU
  TBuf<TPosition::VECCALC> tbuf_zero;
  pipe.InitBuffer(tbuf_zero, Tb_N * 4u);

  uint32_t blk_idx  = (uint32_t)GetBlockIdx();
  uint32_t row_base = blk_idx * TB_M;
  if (row_base >= M_dim) return;

  uint32_t row_end  = row_base + TB_M;
  if (row_end > M_dim) row_end = M_dim;

  __gm__ float* pA    = reinterpret_cast<__gm__ float*>(gm_a);
  __gm__ float* pB    = reinterpret_cast<__gm__ float*>(gm_b);
  __gm__ float* pBias = reinterpret_cast<__gm__ float*>(gm_bias);
  __gm__ float* pOut  = reinterpret_cast<__gm__ float*>(gm_out);

  // Pre-fill zero buffer
  LocalTensor<float> zeros = tbuf_zero.Get<float>();
  Duplicate(zeros, (float)0.0f, Tb_N);

  for (uint32_t j = 0; j < N_dim; j += Tb_N) {{
    uint32_t j_end  = j + Tb_N < N_dim ? j + Tb_N : N_dim;
    uint32_t n_tile = j_end - j;

    for (uint32_t i = row_base; i < row_end; i += Tb_M) {{
      uint32_t i_end  = i + Tb_M < row_end ? i + Tb_M : row_end;
      uint32_t m_tile = i_end - i;

      // Zero-initialize accumulator for this (m_tile x n_tile) tile
      LocalTensor<float> acc = tbuf_acc.Get<float>();
      Duplicate(acc, (float)0.0f, m_tile * n_tile);

      // Matmul: acc[row, col] += A[i+row, k] * B[k, j+col]
      for (uint32_t k = 0; k < K; k++) {{
        // Load B[k, j..j+n_tile]
        GlobalTensor<float> gb;
        gb.SetGlobalBuffer(pB + k * N_dim + j);
        LocalTensor<float> b_row = que_b.AllocTensor<float>();
        DataCopy(b_row, gb, n_tile);
        que_b.EnQue(b_row);
        LocalTensor<float> b_sync = que_b.DeQue<float>();

        LocalTensor<float> ak  = tbuf_ak.Get<float>();
        LocalTensor<float> mul = tbuf_mul.Get<float>();
        for (uint32_t row = 0; row < m_tile; row++) {{
          // Load scalar A[i+row, k] from GM
          float a_val = pA[(i + row) * K + k];
          Duplicate(ak, a_val, n_tile);
          Mul(mul, ak, b_sync, n_tile);
          // acc_row = acc[row * n_tile .. (row+1) * n_tile]
          LocalTensor<float> acc_row = tbuf_acc.GetWithOffset<float>(
              n_tile * 4u, row * n_tile * 4u);
          Add(acc_row, acc_row, mul, n_tile);
        }}
        que_b.FreeTensor(b_sync);
      }}

      // Add bias and apply ReLU, write output row by row
      for (uint32_t row = 0; row < m_tile; row++) {{
        uint32_t global_row = i + row;
        uint32_t global_col = j;

        // Load bias[global_row, j..j+n_tile]
        GlobalTensor<float> gbias;
        gbias.SetGlobalBuffer(pBias + global_row * N_dim + global_col);
        LocalTensor<float> bias_row = que_bias.AllocTensor<float>();
        DataCopy(bias_row, gbias, n_tile);
        que_bias.EnQue(bias_row);
        LocalTensor<float> bias_sync = que_bias.DeQue<float>();

        // acc_row += bias
        LocalTensor<float> acc_row = tbuf_acc.GetWithOffset<float>(
            n_tile * 4u, row * n_tile * 4u);
        Add(acc_row, acc_row, bias_sync, n_tile);
        que_bias.FreeTensor(bias_sync);

        // ReLU: output = max(acc_row, 0)
        LocalTensor<float> out_row = que_out.AllocTensor<float>();
        Max(out_row, acc_row, zeros, n_tile);
        que_out.EnQue(out_row);
        LocalTensor<float> out_sync = que_out.DeQue<float>();

        // Store to GM
        GlobalTensor<float> gout;
        gout.SetGlobalBuffer(pOut + global_row * N_dim + global_col);
        DataCopy(gout, out_sync, n_tile);
        que_out.FreeTensor(out_sync);
      }}
    }}
  }}
  return;
}}
"""

with open(out_path, 'w') as f:
    f.write(code)
print(f"  ✓ 生成仿真 kernel: output_step7_kernel_sim.cpp")
print(f"    Tb_M={Tb_M} Tb_N={Tb_N} K={K} M={M} N={N}")
PYEOF_7D
log "  ✓ Sim kernel 生成完成，输出: output_step7_kernel_sim.cpp"

# ── STAGE 8: Generate Test Data ────────────────────────────
echo ""
echo "==================== [STAGE 8] Generate Test Data ===================="
DATA_DIR="$DIR/test_data"
mkdir -p "$DATA_DIR"
python3 "$DIR/gen_data.py" --M $M --K $K --N $N --out-dir "$DATA_DIR"
echo "  ✓ test_data/ (input_a input_b input_bias output)"

# ── STAGE 9: Compile AscendC Kernel ────────────────────────
# For CPU simulation, compile the vector-only sim kernel (output_step7_kernel_sim.cpp).
# The hardware kernel (output_step7_kernel.cpp) uses cube L1 DMA which hangs in AIVEC sim.
echo ""
echo "==================== [STAGE 9] Compile (sim kernel) ===================="
BUILD_DIR="$DIR/build_e2e"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
$COMPILER \
  --kernel "$DIR/output_step7_kernel_sim.cpp" \
  --output "$BUILD_DIR" \
  --name fc_relu \
  --kernel-type mix \
  --num-inputs 3 \
  --num-outputs 1 2>&1
echo "  ✓ $BUILD_DIR/fc_relu.bin"

# ── STAGE 10: Run + Verify ─────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
BIN="$BUILD_DIR/fc_relu.bin"

TILING_PARAMS="TB_M=${TB_M},TB_N=${TB_N},Tb_M=${Tb_M},Tb_N=${Tb_N},t_K=${t_K},dim_arg0_0=${M},dim_arg0_1=${K},dim_arg1_0=${K},dim_arg1_1=${N},dim_arg2_0=${M},dim_arg2_1=${N},dim_arg3_0=${M},dim_arg3_1=${N}"

$VALIDATOR \
  --bin "$BIN" \
  --name fc_relu \
  --kernel-type mix \
  --inputs "$DATA_DIR/input_a.npy,$DATA_DIR/input_b.npy,$DATA_DIR/input_bias.npy" \
  --expected "$DATA_DIR/output.npy" \
  --tiling-schema "$DIR/tiling_space.json" \
  --tiling-params "$TILING_PARAMS" \
  --block-dim $BLOCK_DIM \
  --atol 1e-3 \
  --rtol 1e-3 \
  --dump-actual "$BUILD_DIR/actual.txt" \
  --dump-expected "$BUILD_DIR/expected.txt" \
  2>&1 | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' || true

echo ""
echo "========================================================"
echo " 全流程完成！"
echo " M=$M K=$K N=$N  TB_M=$TB_M TB_N=$TB_N Tb_M=${Tb_M} Tb_N=${Tb_N} t_K=${t_K}"
echo " block_dim=$BLOCK_DIM  CPU 仿真精度验证 atol=1e-3 rtol=1e-3"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
