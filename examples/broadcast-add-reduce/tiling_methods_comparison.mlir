// ================================================================
//  MLIR Transform Dialect Tiling 两种参数化方式深度对比
//  基于 LLVM 21.x 实际 API
//
//  方式A: transform.param.constant + SSA参数传入 tile_using_for
//  方式B: transform.func.add_index_args + 函数参数化
//
//  核心问题：tiling参数从哪里来？在哪个阶段决定？
// ================================================================


// ================================================================
// 第一节：基本概念澄清
// ================================================================

// Transform Dialect 有三种值类型，必须先搞清楚：
//
//   !transform.any_op          → Handle，指向 payload IR 中的 Op
//   !transform.param<i64>      → Param，纯编译期整数常量（不存在于payload IR中）
//   !transform.any_value       → Value Handle，指向 payload IR 中的 SSA Value
//
// 关键区分：
//   transform.param.constant   → 产生 !transform.param<i64>，是Transform IR层的常量
//   arith.constant (payload)   → 产生 payload IR 中的 index 值（真实代码里的常量）
//
// tile_using_for 接受两种tile size输入（LLVM 21实际签名）：
//   静态：tile_sizes 属性         [128, 64]       编译期固定整数
//   动态：dynamic_sizes 操作数    %sz0, %sz1      可以是 param 或 SSA handle


// ================================================================
// 第二节：方式A —— transform.param.constant 参数化
// ================================================================

// ── A1. 核心语义 ────────────────────────────────────────────────
//
//   transform.param.constant 产生的是 Transform IR 层的"元参数"
//   它在 Transform 解释器运行时（即编译期）被求值
//   不会生成任何 payload IR 代码（不是 arith.constant）
//
//   tile_using_for 接受 !transform.param<i64> 作为 dynamic_sizes
//   → 将参数值注入到生成的 scf.for 的 step 中（作为 arith.constant 或 index_cast）

module attributes {transform.with_named_sequence} {
  transform.named_sequence @tiling_with_param_constant(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // ── param.constant 的正确用法 ────────────────────────────
    // 产生 Transform IR 层的参数，类型是 !transform.param<i64>
    // 注意：0 在这里不是"符号化"的意思，就是字面值0
    // 用0表示"跳过该轴不切"（tile_using_for的约定：0=不切）
    %tile_TB = transform.param.constant 128 : i64   // TB级，128个元素
    %tile_Tb = transform.param.constant 64  : i64   // Tb级，64个元素
    %tile_t  = transform.param.constant 16  : i64   // t级，向量化粒度

    // ── 用 param 驱动 tile_using_for ─────────────────────────
    // LLVM 21 实际语法（注意 dynamic_sizes 关键字）：
    %tiled_TB, %loop_TB =
        transform.structured.tile_using_for %generic
            dynamic_sizes [%tile_TB]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    %tiled_Tb, %loop_Tb =
        transform.structured.tile_using_for %tiled_TB
            dynamic_sizes [%tile_Tb]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    %tiled_t, %loop_t =
        transform.structured.tile_using_for %tiled_Tb
            dynamic_sizes [%tile_t]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// ── A2. param.constant 生成的 payload IR ─────────────────────────
//
// 对于静态shape（M=1024），param.constant=128 会生成：
//
//   %c128 = arith.constant 128 : index   ← 折叠成编译期常量
//   scf.for %i = 0 to 1024 step 128 {
//     ...
//   }
//
// 对于动态shape（M=?），param.constant=128 会生成：
//
//   %c128 = arith.constant 128 : index
//   %dim  = tensor.dim %input, %c0 : tensor<?xf32>
//   scf.for %i = 0 to %dim step 128 {   ← step是常量，bound是动态的
//     ...
//   }
//
// 关键特性：step（分块大小）是编译期固定的，只有loop bound是动态的


// ── A3. param.constant 真正的用途：条件分支选择 ──────────────────

module attributes {transform.with_named_sequence} {
  transform.named_sequence @param_for_dispatch(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // 从Op中提取shape信息作为 param（这是param的核心价值！）
    // transform.structured.match_structured.get_loop_range 等
    // → 得到 !transform.param<i64> 类型的shape信息

    // 基于 param 做条件判断，选择不同的tiling策略
    %size_M = transform.structured.match_operand_type_rank %generic {dim = 0}
              : (!transform.any_op) -> !transform.param<i64>

    // 与阈值比较
    %threshold = transform.param.constant 512 : i64
    transform.match.param.cmpi ult %size_M, %threshold
        : !transform.param<i64>
    // 如果 M < 512，走小矩阵策略（否则 silenceable failure，走备选）

    %tile_small = transform.param.constant 32 : i64
    %tiled, %loop = transform.structured.tile_using_for %generic
        dynamic_sizes [%tile_small]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}


// ================================================================
// 第三节：方式B —— transform.func.add_index_args（你提到的第二种）
// ================================================================

// ── B0. 重要澄清：你原始写法的问题 ──────────────────────────────
//
// 你写的：
//   %func_new, %TB_M, %Tb_M = transform.func.add_index_args %func, 2
//
// LLVM 21 中 transform.func 下没有 add_index_args 这个 Op！
// 正确的相关 Op 是：
//   transform.func.cast_and_call
//   transform.func.replace_func_signature
//   transform.func.deduplicate_func_args
//
// 你想要的效果（给函数增加index参数用于运行时tiling）
// 需要用 replace_func_signature 或者完全不同的方式实现
//
// 下面演示最接近你意图的 LLVM 21 实际写法：


// ── B1. 正确方式：通过函数签名传入运行时tiling参数 ──────────────
//
// 核心思想：不在Transform Dialect里传参，
//           而是在 payload IR 的函数签名里加 index 参数，
//           这些参数在 runtime 由 host 的 tiling_func 计算后传入

// payload IR：函数已经有了 tiling 参数
func.func @kernel_with_runtime_tiling(
    %input  : tensor<?xf32>,
    %output : tensor<?xf32>,
    // ↓ 这些是 runtime tiling 参数（host计算后传入）
    %TB_size : index,
    %Tb_size : index,
    %t_size  : index
) -> tensor<?xf32> {
  // 函数体先省略，下面的Transform脚本会对它做tiling
  %result = linalg.generic {
    indexing_maps  = [affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<?xf32>) outs(%output : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.log %in : f32
    linalg.yield %v : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// Transform 脚本：用函数参数（运行时值）驱动 tiling
module attributes {transform.with_named_sequence} {
  transform.named_sequence @tiling_with_func_args(
      %root : !transform.any_op
  ) {
    %func = transform.structured.match
        ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %generic = transform.structured.match
        ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op

    // ── 提取函数参数作为 Value Handle ────────────────────────
    // transform.get_operand 获取 func 的第 N 个参数
    // 返回 !transform.any_value（payload IR 中的 SSA value）
    %TB_val = transform.get_operand %func[2]   // 第2个参数 = %TB_size
              : (!transform.any_op) -> !transform.any_value
    %Tb_val = transform.get_operand %func[3]   // 第3个参数 = %Tb_size
              : (!transform.any_op) -> !transform.any_value
    %t_val  = transform.get_operand %func[4]   // 第4个参数 = %t_size
              : (!transform.any_op) -> !transform.any_value

    // ── 用 Value Handle 驱动 tile_using_for ──────────────────
    // 此时 step 是真正的 SSA value（运行时确定）
    // 生成的 scf.for 的 step 会是这个 SSA value，而非常量
    %tiled_TB, %loop_TB =
        transform.structured.tile_using_for %generic [%TB_val]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    %tiled_Tb, %loop_Tb =
        transform.structured.tile_using_for %tiled_TB [%Tb_val]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    %tiled_t, %loop_t =
        transform.structured.tile_using_for %tiled_Tb [%t_val]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// ── B2. 方式B 生成的 payload IR ──────────────────────────────────
//
// 生成的 scf.for 的 step 是函数参数（SSA value）：
//
//   func.func @kernel_with_runtime_tiling(
//       %input, %output,
//       %TB_size: index,   ← 运行时传入
//       %Tb_size: index,
//       %t_size : index
//   ) {
//     %dim = tensor.dim %input, %c0
//     scf.for %i = 0 to %dim step %TB_size {          ← step=运行时值！
//       scf.for %j = 0 to %TB_size step %Tb_size {    ← step=运行时值！
//         scf.for %k = 0 to %Tb_size step %t_size {   ← step=运行时值！
//           ...
//         }
//       }
//     }
//   }
//
// 关键特性：step 是运行时动态值，编译器无法做常量折叠/循环展开


// ================================================================
// 第四节：两种方式的本质差异对比
// ================================================================

// ┌─────────────────────────┬──────────────────────┬──────────────────────┐
// │ 维度                    │ 方式A: param.constant│ 方式B: 函数参数Value │
// ├─────────────────────────┼──────────────────────┼──────────────────────┤
// │ 参数类型                │!transform.param<i64> │!transform.any_value  │
// │ 参数在哪个IR层          │Transform IR层（元层）│Payload IR层（真实层）│
// │ 参数何时确定            │Transform解释器运行时 │Kernel执行时(runtime) │
// │                         │（即编译期）          │                      │
// │ 生成的scf.for step      │arith.constant（常量）│ SSA value（动态）    │
// │ 编译器能否折叠/展开     │ ✅ 能                │ ❌ 不能（动态值）    │
// │ 支持不同shape用同一编译 │ ❌ 一个参数值一个编译│ ✅ 一次编译多shape   │
// │ 结果                    │                      │                      │
// │ 是否支持真正动态shape   │ 部分（bound动态，    │ ✅ 完全动态          │
// │                         │ step静态）           │                      │
// │ 对应你们方案的哪个阶段  │ 编译期模板生成       │ Runtime tiling参数   │
// │ 适合场景                │ 编译期已知最优tile   │ Runtime选择最优tile  │
// │ 性能优化潜力            │ 更高（静态可展开）   │ 稍低（动态步长）     │
// │ 灵活性                  │ 低（需重编译换参数） │ 高（同Kernel换参数） │
// └─────────────────────────┴──────────────────────┴──────────────────────┘


// ================================================================
// 第五节：结合你们的 AutoFuse 方案，应该用哪种？
// ================================================================

// 你们的核心需求：
//   编译期：生成 N 个 Kernel 模板（对应 N 种轴切分结构）
//   运行期：拿到具体 shape → tiling_func → 选 Kernel + 传参数
//
// → 这是一个混合模式，两种方式都需要！

// ── 推荐方案：分层使用 ──────────────────────────────────────────
//
// 层次1：编译期 - 用 param.constant 决定 Kernel "模板结构"
//         即：几层循环？轴的顺序？是否向量化？
//         这部分固定在 Kernel 代码里
//
// 层次2：运行期 - 用函数参数（Value Handle方式）传入具体tile大小
//         即：TB_size=多少？Tb_size=多少？
//         这部分由 tiling_func 在 host 计算后作为参数传入

// ── 完整示例：Matmul+Add+Log 的混合方案 ─────────────────────────

// Step 1: payload IR 函数签名（包含运行时tiling参数）
func.func @fused_matmul_add_log(
    %A       : memref<?x?xf16>,
    %B       : memref<?x?xf16>,
    %Bias    : memref<?xf16>,
    %Out     : memref<?x?xf16>,
    // ── 运行时tiling参数（由host tiling_func计算后传入）──
    %M_tb    : index,     // 每核M行数
    %d0_Tb   : index,     // Matmul M块（编译期通常固定128，但可动态）
    %d1_Tb   : index,     // Matmul N块
    %d2_Tb   : index,     // Matmul K块
    %vec_Tb  : index      // Vector段行数
) {
  // 函数体：包含 linalg.matmul 和 linalg.generic(add+log)
  // 由 Transform 脚本对这些 Op 做 tiling
  return
}

// Step 2: Transform 脚本（编译期决定结构，运行期决定参数值）
module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_matmul_schedule(
      %root : !transform.any_op
  ) {
    %func = transform.structured.match
        ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // ── 提取运行时tiling参数（Value Handle）────────────────
    %v_M_tb   = transform.get_operand %func[4]
                : (!transform.any_op) -> !transform.any_value
    %v_d0_Tb  = transform.get_operand %func[5]
                : (!transform.any_op) -> !transform.any_value
    %v_d1_Tb  = transform.get_operand %func[6]
                : (!transform.any_op) -> !transform.any_value
    %v_d2_Tb  = transform.get_operand %func[7]
                : (!transform.any_op) -> !transform.any_value
    %v_vec_Tb = transform.get_operand %func[8]
                : (!transform.any_op) -> !transform.any_value

    // ── 匹配 Matmul ────────────────────────────────────────
    %matmul = transform.structured.match
        ops{["linalg.matmul"]} in %func
        : (!transform.any_op) -> !transform.any_op

    // ── TB级: 核间切分（沿M轴，step=运行时值M_tb）──────────
    //    用 Value Handle 传入运行时参数
    %matmul_TB, %loop_M =
        transform.structured.tile_using_for %matmul [%v_M_tb, 0, 0]
        // [M轴切M_tb, N轴不切=0, K轴不切=0]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    // ── 编译期决定的结构：K轴循环（step用param.constant固定）
    //    K块大小 64 在编译期确定，硬件对齐约束
    %k_tile = transform.param.constant 64 : i64
    %matmul_K, %loop_K =
        transform.structured.tile_using_for %matmul_TB [0, 0, %k_tile]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    // → 生成的 scf.for step = arith.constant 64（编译期常量，可展开优化）

    // ── 匹配 Vector 段 (add+log) ───────────────────────────
    %vector_op = transform.structured.match
        ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op

    // ── Vector段TB切分（跟随Matmul的M轴，step=运行时值）──────
    %vec_TB, %vec_loop =
        transform.structured.tile_using_for %vector_op [%v_M_tb]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    // ── t级向量化（编译期固定128，用 param.constant）─────────
    //    向量宽度由硬件决定，编译期固定，允许向量化优化
    %t_tile = transform.param.constant 128 : i64
    transform.structured.vectorize %vec_TB
        vector_sizes [%t_tile]
        : !transform.any_op

    transform.yield
  }
}

// ── 生成的 payload IR 对比 ─────────────────────────────────────
//
// 上述混合方案生成：
//
//   func.func @fused_matmul_add_log(%A, %B, %Bias, %Out,
//       %M_tb: index, %d0_Tb: index, ...
//   ) {
//     // TB级：M_tb 是函数参数（运行时值），step动态
//     scf.for %m = 0 to %M step %M_tb {          ← 动态step
//
//       // K级：64 是编译期常量，编译器可展开
//       scf.for %k = 0 to %K step 64 {           ← 静态step ✅可展开
//         ascend.cube.mmad ...
//       }
//
//       // Vector t级：128 是编译期常量，向量化指令直接生成
//       vector.transfer_read ...                  ← 静态向量长度 ✅
//       arith.addf ... : vector<128xf16>
//       math.log  ... : vector<128xf16>
//     }
//   }


// ================================================================
// 第六节：LLVM 21 中 tile_using_for 的完整签名（澄清混淆点）
// ================================================================

// LLVM 21 实际 Op 定义（来自 LinalgTransformOps.td）：
//
// def TileUsingForOp : Op<Transform_Dialect,
//                         "structured.tile_using_for", ...> {
//   let arguments = (ins
//     TransformHandleTypeInterface:$target,
//     // 静态sizes属性（编译期字面量）
//     DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
//     // 动态sizes操作数（可以是 param 或 value handle）
//     Variadic<Transform_AnyParamOrValueType>:$dynamic_sizes,
//     // 循环interchange
//     DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$interchange,
//     // 尾块处理策略
//     DefaultValuedAttr<...>:$scalable_sizes
//   );
// }
//
// 合法的调用形式：
//
// 形式1：纯静态属性（最常见，编译期常量）
//   %tiled, %loop = transform.structured.tile_using_for %op tile_sizes [128, 64]
//                   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//
// 形式2：混合（静态+动态，0表示用动态值替代）
//   %tiled, %loop = transform.structured.tile_using_for %op tile_sizes [0, 64]
//                   dynamic_sizes [%runtime_M]
//                   : (!transform.any_op, !transform.any_value)
//                  -> (!transform.any_op, !transform.any_op)
//   // 轴0用运行时值，轴1用64
//
// 形式3：纯动态（你们需要的方式）
//   %tiled, %loop = transform.structured.tile_using_for %op [%sz0, %sz1]
//                   : (!transform.any_op, !transform.any_value, !transform.any_value)
//                  -> (!transform.any_op, !transform.any_op, !transform.any_op)
//
// 形式4：param（编译期可知的符号参数）
//   %p = transform.param.constant 128 : i64
//   %tiled, %loop = transform.structured.tile_using_for %op [%p]
//                   : (!transform.any_op, !transform.param<i64>)
//                  -> (!transform.any_op, !transform.any_op)


// ================================================================
// 第七节：你们 AutoFuse 方案的完整 Transform 脚本建议
// ================================================================

// 基于 Matmul+Add+Log 的完整可运行 Transform 脚本（LLVM 21）：

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %module : !transform.any_op {transform.readonly}
  ) {

    // 1. 匹配目标函数
    %func = transform.structured.match
        ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    // 2. 匹配 Matmul 和 Generic（Add+Log）
    %matmul = transform.structured.match
        ops{["linalg.matmul"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %generic = transform.structured.match
        ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op

    // 3. 提取运行时参数 handle（方式B）
    //    假设函数签名：@kernel(%A, %B, %Bias, %Out, %M_tb, %d2_Tb)
    %h_M_tb  = transform.get_operand %func[4]
               : (!transform.any_op) -> !transform.any_value
    %h_d2_Tb = transform.get_operand %func[5]
               : (!transform.any_op) -> !transform.any_value

    // 4. Matmul TB级切分（M轴，动态参数）
    %mm_TB, %loop_M =
        transform.structured.tile_using_for %matmul tile_sizes [0, 0, 0]
        // tile_sizes全0，通过下面的dynamic_sizes传入
        dynamic_sizes [%h_M_tb]
        : (!transform.any_op, !transform.any_value)
       -> (!transform.any_op, !transform.any_op)

    // 5. Matmul K轴切分（编译期固定64，方式A）
    //    K块大小对Cube硬件有对齐约束，编译期固定更有利于代码生成
    %mm_K, %loop_K =
        transform.structured.tile_using_for %mm_TB tile_sizes [0, 0, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 6. 向量化 Generic（Add+Log），t=128固定（方式A）
    %p_vec = transform.param.constant 128 : i64
    transform.structured.vectorize %generic
        vector_sizes [%p_vec, %p_vec]
        : !transform.any_op

    // 7. 映射TB循环到 AiCore（相当于 blockIdx）
    transform.loop.map_to_blocks %loop_M
        {block_dims = [0]}
        : (!transform.any_op) -> ()

    // 8. Bufferize + 标准lower
    %bufferized = transform.bufferization.one_shot_bufferize %func
        {bufferize_function_boundaries = true}
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// ================================================================
// 总结决策树
// ================================================================
//
// 你要用哪种方式？
//
//   tile size 在编译期已知（对齐到硬件，如K轴=64）？
//   └─→ 用 tile_sizes [64]（静态属性）或 param.constant
//       优点：编译器可常量折叠、循环展开、向量化
//
//   tile size 在运行期确定（随shape变化，如TB_M=M/core_num）？
//   └─→ 用函数参数 + transform.get_operand + Value Handle
//       优点：一个Kernel处理任意shape
//
//   tile size 取决于shape但有有限几种取值（如bucket）？
//   └─→ 用 transform.alternatives / transform.match.param.cmpi
//       生成多个专化版本 + runtime dispatch
//       这最接近你们"N个Kernel模板"的设计！
//
// 你们方案的最佳实践（对应AutoFuse的TB/Tb/t）：
//   TB_size：运行期（Value Handle）→ 随核数/shape变化
//   Tb_size：运行期（Value Handle）→ 随UB容量/shape变化
//   t_size ：编译期（静态属性128）→ 硬件向量宽度固定
//   K_size ：编译期（静态属性64） → Cube对齐约束固定
