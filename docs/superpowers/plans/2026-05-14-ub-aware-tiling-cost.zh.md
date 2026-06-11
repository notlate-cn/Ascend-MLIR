# UB-Aware Tiling Cost — 修复 dyn-bucketed-e2e 大 R 全零 输出

> **For agentic workers:** REQUIRED SUB-SKILL: 使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 按任务逐条实施。所有步骤用 checkbox (`- [ ]`) 标记。

**Goal:** 让 `network_runner` 的 phase-3 默认 picker 和 `autotuner` 在选 XBLOCK/XBLOCK_SUB 等候选时,根据 lowered-IR 中所有 `ascendc.pipe.init_buffer` / `init_queue` 大小累加得到的 **符号化 UB cost 表达式** 进行剪枝,使任何会让 UB 占用 > SoC 预算的候选都被剔除。`examples/dyn-bucketed-e2e/run.sh` 在 R=512(以及更大 R)上不再产出全零。

**Architecture:** 三层串联:
1. **codegen 层(C++)**:`CannTranslation` 在写 `<kernel>__v<i>_space.json` 时,walk 函数体上所有 init_buffer / init_queue 操作,把每个 size operand 的 `arith.muli/addi/divi/constant/emitasc.member` 反向求图翻成 `SymExpr`,32B 对齐后求和,emit 为新字段 `ub_cost_bytes_expr`(字符串,和现有 `axis_extent_expr` 语法一致)。同时 emit `ub_budget_bytes`(从 SoC 表查表得到的整数)。
2. **picker 层(Python)**:`python/network_runner.py` 在选 tunable 候选时,evaluate `ub_cost_bytes_expr`,把会超 budget 的候选剔除;候选全空时报 `PickerError`,提示 "需要 RBLOCK 切分 / 减少 reduce extent / 提升 budget"。
3. **autotuner 层(C++)**:`tools/autotuner/autotuner_main.cpp` 在 enumerate 候选后,evaluate 同一表达式,push 剪枝计数,与现有 "inner-tile > outer-tile" 剪枝逻辑并列。

**Tech Stack:** MLIR, C++ (LLVM ADT, `llvm::json`), Python 3, `mlir::afir::symshape::SymExpr`, 现有 `eval_block_dim` 表达式 evaluator。

**Non-goals:** 不实现 R 轴 RBLOCK 切分(选项 Z,留单独 plan);不修 autotuner false-pass(独立 bug,后续 follow-up)。

---

## File Structure

| Path | 作用 |
|---|---|
| `include/Target/CannKernel/SocSpec.h` *(新)* | SoC → `{TOTAL_UB_SIZE, TOTAL_VEC_LOCAL_SIZE, ONE_BLK_SIZE}` 常量映射,数据来自 CANN `kernel_utils_constants.h` |
| `include/Target/CannKernel/UbCostExpr.h` *(新)*, `lib/Target/CannKernel/UbCostExpr.cpp` *(新)* | `Optional<SymExpr> liftSizeOperand(Value)` 把 arith DAG 翻成 SymExpr;`SymExpr align32(...)` 32B 对齐 |
| `lib/Target/CannKernel/CannTranslation.cpp` *(改)* | (a) emit `ub_budget_bytes` (b) walk init_buffer/init_queue,提取每个 size 的 SymExpr → emit `ub_cost_bytes_expr` |
| `python/network_runner.py` *(改)* | picker 用 `ub_cost_bytes_expr` 剪枝;`PickerError` 类 |
| `tools/autotuner/autotuner_main.cpp` *(改)* | 在 enumerate 后按 UB cost 剪枝 |
| `examples/dyn-bucketed-e2e/run_R_sweep.sh` *(新)* | R={64,128,256,512,1024} sweep 脚本 |
| `examples/dyn-bucketed-e2e/BUG_REPORT.md` *(改)* | 标记 fixed-by-y,引到本 plan |
| `test/CannTranslation/ub-cost-expr.mlir` *(新)* | filecheck 测试:emit 的 ub_cost_bytes_expr 字面值匹配 |

设计说明:`UbCostExpr.{h,cpp}` 独立出来是因为 IR→SymExpr 的反向求图逻辑复用价值高(后续 RBLOCK 设计也会用),并且单独编译/测试方便。

---

## Task 1: SocSpec.h — SoC UB 常量表

**Files:**
- Create: `include/Conversion/CannTranslation/SocSpec.h`

数据来源:`/home/gser/Ascend/cann-9.0.0/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h`。NPU_ARCH 与 SoC 字符串对应根据 `--soc` 已有的命名约定(`Ascend910B1`,`Ascend910A`,`Ascend310B` 等)。

- [ ] **Step 1: 写 header**

```cpp
//===- SocSpec.h - SoC capacity constants from CANN ---------*- C++ -*-===//
#ifndef AFIR_CONVERSION_CANNTRANSLATION_SOCSPEC_H
#define AFIR_CONVERSION_CANNTRANSLATION_SOCSPEC_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir::afir::canntranslation {

struct SocSpec {
  uint32_t totalUbSize;       // TOTAL_UB_SIZE (raw HW)
  uint32_t totalVecLocalSize; // TOTAL_VEC_LOCAL_SIZE (user TBuf/TQue pool)
  uint32_t oneBlockSize;      // ONE_BLK_SIZE (allocator alignment)
};

/// Returns the SoC capacity record for `soc`, or nullopt if unknown.
/// Source: CANN asc/impl/basic_api/utils/kernel_utils_constants.h.
inline std::optional<SocSpec> getSocSpec(llvm::StringRef soc) {
  // NPU_ARCH 2201 — Ascend910B1
  if (soc == "Ascend910B1")
    return SocSpec{192u * 1024, 184u * 1024, 32u};
  // NPU_ARCH 1001 / 2002 — Ascend910 / Ascend910A
  if (soc == "Ascend910" || soc == "Ascend910A")
    return SocSpec{256u * 1024, 248u * 1024, 32u};
  // NPU_ARCH 3002 — Ascend910C
  if (soc == "Ascend910C")
    return SocSpec{248u * 1024, 184u * 1024, 32u};
  // NPU_ARCH 3102
  if (soc == "Ascend910N")
    return SocSpec{256u * 1024, 184u * 1024, 32u};
  // NPU_ARCH 3510 / 5102 — Ascend910D / next gen
  if (soc == "Ascend910D")
    return SocSpec{248u * 1024, 248u * 1024, 32u};
  // NPU_ARCH 3003 / 3113 — Ascend310B
  if (soc == "Ascend310B")
    return SocSpec{118u * 1024, 118u * 1024, 32u};
  return std::nullopt;
}

} // namespace mlir::afir::canntranslation

#endif // AFIR_CONVERSION_CANNTRANSLATION_SOCSPEC_H
```

- [ ] **Step 2: 验证 header 自洽编译**

```bash
echo '#include "Conversion/CannTranslation/SocSpec.h"
int main(){auto s = mlir::afir::canntranslation::getSocSpec("Ascend910B1");
return s ? 0 : 1;}' > /tmp/check_socspec.cpp
g++ -Iinclude -c /tmp/check_socspec.cpp -o /tmp/check_socspec.o
echo "compiled: $?"
```

Expected: `compiled: 0`

- [ ] **Step 3: Commit**

```bash
git add include/Conversion/CannTranslation/SocSpec.h
git commit -m "feat(canntranslation): add SocSpec table for UB capacity by SoC

Mirrors the per-NPU_ARCH constants from CANN
asc/impl/basic_api/utils/kernel_utils_constants.h.  Used in subsequent
commits to expose ub_budget_bytes in tiling_space.json so the picker /
autotuner can prune candidates that overflow UB.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: UbCostExpr — IR size operand → SymExpr lift

**Files:**
- Create: `include/Conversion/CannTranslation/UbCostExpr.h`
- Create: `lib/Target/CannKernel/UbCostExpr.cpp`

这一块是 IR 反向求图。可被遇到的 op 类型(来自 `kernel_group0_lowered.mlir` 实测):
- `arith.constant` → `SymExpr::constant(value)`
- `arith.muli`, `arith.addi`, `arith.subi`, `arith.divui`, `arith.divsi` → 对应 SymExpr binop
- `emitasc.member %tilingData "<name>" : !emitasc.py_struct<...>, i64` → `SymExpr::sym(SymId)`,其中 SymId 来自一个 `StringRef name → SymId` 表(picker/autotuner 那边再用 name 反查)
- `arith.index_cast`(从 i64 → index 之间)→ 透传内部 expr
- `func.func` block arg 是 `!emitasc.py_struct` 的字段访问入口,通过 `emitasc.member` 间接消费

任何不在白名单的 op → 返回 `std::nullopt`,调用方记录 "ub_cost_bytes_expr 不可用",picker 见 null 就不剪枝(降级到当前行为)。

- [ ] **Step 1: 写 UbCostExpr.h**

```cpp
//===- UbCostExpr.h - lift init_buffer size operand to SymExpr -*- C++ -*-===//
#ifndef AFIR_CONVERSION_CANNTRANSLATION_UBCOSTEXPR_H
#define AFIR_CONVERSION_CANNTRANSLATION_UBCOSTEXPR_H

#include "Analysis/SymbolicShape/SymExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

namespace mlir::afir::canntranslation {

/// Bidirectional name <-> SymId table for lifting `emitasc.member`-typed
/// leaves.  Caller owns the lifetime.
struct NameSymTable {
  llvm::StringMap<symshape::SymId> nameToId;
  llvm::DenseMap<symshape::SymId, std::string> idToName;
  symshape::SymId getOrCreate(llvm::StringRef name);
};

/// Walks the def-use chain of `sizeOperand` and lifts it to a SymExpr.  Returns
/// nullopt on the first unknown op kind (callers downgrade to "no UB budget
/// known").  Supported ops: arith.{constant,muli,addi,subi,divui,divsi},
/// arith.index_cast, emitasc.member (leaf).  `names` is appended to as
/// emitasc.member leaves are encountered.
std::optional<symshape::SymExpr> liftSizeOperand(mlir::Value sizeOperand,
                                                 NameSymTable &names);

/// align_up(s, 32) expressed in SymExpr:
///   ceilDiv(s, 32) * 32.
symshape::SymExpr align32(symshape::SymExpr s);

} // namespace mlir::afir::canntranslation

#endif
```

- [ ] **Step 2: 写 UbCostExpr.cpp**

```cpp
//===- UbCostExpr.cpp -----------------------------------------------------===//
#include "Conversion/CannTranslation/UbCostExpr.h"
#include "Dialect/EmitAsc/IR/EmitAscOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::afir;
namespace ss = mlir::afir::symshape;

namespace mlir::afir::canntranslation {

ss::SymId NameSymTable::getOrCreate(llvm::StringRef name) {
  auto it = nameToId.find(name);
  if (it != nameToId.end()) return it->second;
  ss::SymId id = static_cast<ss::SymId>(nameToId.size());
  nameToId.try_emplace(name, id);
  idToName[id] = name.str();
  return id;
}

std::optional<ss::SymExpr> liftSizeOperand(Value v, NameSymTable &names) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp) return std::nullopt;

  if (auto c = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ss::SymExpr::constant(ia.getInt());
    return std::nullopt;
  }
  if (auto mb = dyn_cast<emitasc::MemberOp>(defOp)) {
    // emitasc.member %tilingData "<name>" — leaf SymExpr keyed by field name.
    std::string field = mb.getName().str();
    return ss::SymExpr::sym(names.getOrCreate(field));
  }
  if (auto ic = dyn_cast<arith::IndexCastOp>(defOp))
    return liftSizeOperand(ic.getIn(), names);

  auto binary = [&](auto folder) -> std::optional<ss::SymExpr> {
    auto lhs = liftSizeOperand(defOp->getOperand(0), names);
    auto rhs = liftSizeOperand(defOp->getOperand(1), names);
    if (!lhs || !rhs) return std::nullopt;
    return folder(*lhs, *rhs);
  };
  if (isa<arith::MulIOp>(defOp))
    return binary([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::mul(a, b); });
  if (isa<arith::AddIOp>(defOp))
    return binary([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::add(a, b); });
  if (isa<arith::SubIOp>(defOp))
    return binary([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::sub(a, b); });
  if (isa<arith::DivUIOp>(defOp) || isa<arith::DivSIOp>(defOp))
    return binary([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::ceilDiv(a, b); }); // round-up safe-side
  return std::nullopt;
}

ss::SymExpr align32(ss::SymExpr s) {
  ss::SymExpr c32 = ss::SymExpr::constant(32);
  return ss::SymExpr::mul(ss::SymExpr::ceilDiv(s, c32), c32);
}

} // namespace
```

注:`DivUI/DivSI` 用 `ceilDiv` 是保守(向上取整),用于代价估算只可能高估,picker 不会因此漏掉真危险的候选。

- [ ] **Step 3: 写 CMake**

修改 `lib/Target/CannKernel/CMakeLists.txt`:把 `UbCostExpr.cpp` 加入 SOURCES。

```bash
sed -n '1,20p' lib/Target/CannKernel/CMakeLists.txt
```

确认有 `CannTranslation.cpp`,然后:

```cmake
# 在 add_mlir_translation_library(...) 的 SOURCES 段加一行:
  UbCostExpr.cpp
```

- [ ] **Step 4: 构建验证**

```bash
cmake --build build -j 8 2>&1 | tail -5
```

Expected: 编译通过,无错。

- [ ] **Step 5: Commit**

```bash
git add include/Conversion/CannTranslation/UbCostExpr.h \
        lib/Target/CannKernel/UbCostExpr.cpp \
        lib/Target/CannKernel/CMakeLists.txt
git commit -m "feat(canntranslation): UbCostExpr lifts init_buffer size to SymExpr

Walks arith / emitasc DAG behind an init_buffer / init_queue size
operand and rebuilds it as a SymExpr keyed on TilingData field names
(XBLOCK_SUB, dim_arg0_2, ...).  Unknown ops short-circuit to nullopt;
no IR mutation.  align32 helper for 32B bump-pointer rounding.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: CannTranslation — emit `ub_cost_bytes_expr` + `ub_budget_bytes`

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp:1991-2100` (the `emitTilingSpaceJson` function)

接入点是 emit 函数中刚生成 `root["axis_extent_expr"] = axisExtentExpr;` 之后(约第 2089 行)。

- [ ] **Step 1: 写测试 mlir(failing)**

新建 `test/CannTranslation/ub-cost-expr.mlir`:

```mlir
// RUN: afir-translate %s --emit-cann-kernel --emit-tiling-space-dir=%t \
// RUN:   && FileCheck %s < %t/min_kernel__v0_space.json

// CHECK: "ub_budget_bytes": 188416
// CHECK: "ub_cost_bytes_expr": "(ceildiv((XBLOCK_SUB*dim_arg0_2*4),32)*32)"

module attributes {vector_plan.tiling_infos = [{block_dim_expr = "1",
    fields = [{abi_index = 0 : i32, arg_index = 5 : i32, axis_size = -1 : i64,
               default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}],
    kernel_id = "min_kernel__v0"}]} {
  func.func private @min_kernel__v0(
      %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
      %arg5: !emitasc.py_struct<"TilingData", [i64, i64],
                                ["XBLOCK_SUB", "dim_arg0_2"]>)
      attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %0 = emitasc.member %arg5 "XBLOCK_SUB" : !emitasc.py_struct<...>, i64
    %1 = emitasc.member %arg5 "dim_arg0_2" : !emitasc.py_struct<...>, i64
    %c4 = arith.constant 4 : i64
    %2 = arith.muli %0, %1 : i64
    %3 = arith.muli %2, %c4 : i64
    %4 = arith.index_cast %3 : i64 to index
    %pipe = ascendc.tpipe : <>
    %tbuf = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %pipe, %tbuf, %4 : !ascendc.tbuf<vecin>, index
    func.return
  }
}
```

预期单条 init_buffer,size = `align32(XBLOCK_SUB * dim_arg0_2 * 4)`,emit 字面值匹配 CHECK。

```bash
build/bin/llvm-lit -v test/CannTranslation/ub-cost-expr.mlir
```

Expected: FAIL("ub_cost_bytes_expr": "" or 字段不存在)。

- [ ] **Step 2: 修改 `emitTilingSpaceJson`,walk init_buffer/init_queue**

在文件 head 加 include:

```cpp
#include "Conversion/CannTranslation/UbCostExpr.h"
#include "Conversion/CannTranslation/SocSpec.h"
```

在 `emitTilingSpaceJson` 的 `root["axis_extent_expr"] = axisExtentExpr;` 之后插入:

```cpp
  // ---- UB cost model (Y plan) ------------------------------------------
  // Walk init_buffer / init_queue ops in this func, lift each size operand
  // to SymExpr, align_up to 32B, sum.  If any op's size is non-liftable,
  // fall back to leaving ub_cost_bytes_expr unset (picker/autotuner then
  // skip pruning for this kernel).
  std::string ubCostExpr;
  if (auto spec = getSocSpec("Ascend910B1")) // TODO: thread real --soc
    root["ub_budget_bytes"] = (int64_t)spec->totalVecLocalSize;
  {
    NameSymTable names;
    symshape::SymExpr total;
    bool ok = true;
    funcOp.walk([&](Operation *op) {
      Value sizeOperand;
      if (auto ib = dyn_cast<ascendc::TPipeInitBufferOp>(op))
        sizeOperand = ib.getLen();
      else if (auto iq = dyn_cast<ascendc::TPipeInitQueueOp>(op))
        sizeOperand = iq.getLen();
      else
        return;
      auto e = liftSizeOperand(sizeOperand, names);
      if (!e) { ok = false; return; }
      auto aligned = align32(*e);
      total = total.isValid() ? symshape::SymExpr::add(total, aligned) : aligned;
    });
    if (ok && total.isValid()) {
      auto nameFor = [&](symshape::SymId id) -> std::string {
        auto it = names.idToName.find(id);
        return it != names.idToName.end() ? it->second : "?";
      };
      ubCostExpr = total.emitC(nameFor);
    }
  }
  if (!ubCostExpr.empty())
    root["ub_cost_bytes_expr"] = ubCostExpr;
```

注:`TPipeInitBufferOp::getLen()` / `TPipeInitQueueOp::getLen()` 是 ODS-生成的 getter,确认 op 定义后改成实际名(可能是 `getBufferLen`、`getSize` 等)。若不确定:

```bash
grep -nE "def TPipeInitBufferOp|def TPipeInitQueueOp" include/Dialect/AscendC/IR/*.td
```

找 `let arguments = (ins ... );` 中 length operand 的名字。

- [ ] **Step 3: 重新构建 + 测试 should pass**

```bash
cmake --build build -j 8 2>&1 | tail -3 \
&& build/bin/llvm-lit -v test/CannTranslation/ub-cost-expr.mlir
```

Expected:

```
PASS: AfirMLIR :: CannTranslation/ub-cost-expr.mlir
```

- [ ] **Step 4: 实际 e2e 产物对比**

重跑 dyn-bucketed-e2e codegen,验证生成的 space.json:

```bash
source /home/gser/Ascend/ascend-toolkit/set_env.sh >/dev/null
W=/tmp/dyn_check && rm -rf "$W" && mkdir -p "$W"
python3 examples/dyn-bucketed-e2e/gen_inputs.py --out-dir "$W" --d0 8 --d1 16 --d2 512 >/dev/null
export PATH=$PWD/build/bin:$PATH
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg examples/dyn-bucketed-e2e/model.mlir \
  --inputs $W/{a,b,c,d,e,f,g,init0,init1}.npy \
  --expected $W/expected0.npy $W/expected1.npy \
  --workdir $W --soc Ascend910B1 --max-phase 2 > /dev/null
python3 -c "
import json
s = json.load(open('$W/kernel_group0__v0_space.json'))
print('ub_budget_bytes:', s.get('ub_budget_bytes'))
print('ub_cost_bytes_expr:', s.get('ub_cost_bytes_expr'))"
```

Expected:
```
ub_budget_bytes: 188416
ub_cost_bytes_expr: <非空,含 XBLOCK_SUB 和 dim_arg0_2>
```

- [ ] **Step 5: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp \
        test/CannTranslation/ub-cost-expr.mlir
git commit -m "feat(canntranslation): emit ub_cost_bytes_expr + ub_budget_bytes

For each per-func tiling_space.json, walk init_buffer/init_queue ops,
lift size operands to SymExpr, align-up each to 32B, and sum.  Emit
the symbolic total as ub_cost_bytes_expr alongside ub_budget_bytes
(SoC-table lookup).  Pickers in network_runner.py and the autotuner
will consume these in follow-up commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: network_runner.py — UB-aware picker

**Files:**
- Modify: `python/network_runner.py:75-150` (extend `_eval` grammar to support `ceildiv`)
- Modify: `python/network_runner.py:280-322` (the phase3 default-tile picker)

`_eval` grammar 当前支持 `+ - * /` 和 `ceil(x/y)`(顶层) ;现在需要支持 SymExpr `emitC` 输出的 `ceildiv(a,b)` 调用。可以在 _eval 里加一个 `ceildiv(` 前缀分支,或者预处理替换 `ceildiv(a,b)` → `ceil((a)/(b))`。

- [ ] **Step 1: 加 `ceildiv` 语法**

在 `python/network_runner.py:75` 的 `eval_block_dim` 函数体内 `def _eval(e):` 顶部加:

```python
    def _eval(e: str) -> int:
        e = e.strip()
        # SymExpr emitC uses ceildiv(a,b); rewrite to ceil((a)/(b)) once
        while "ceildiv(" in e:
            i = e.index("ceildiv(")
            # find matching paren
            depth = 0; j = i + len("ceildiv(")
            while j < len(e):
                if e[j] == '(': depth += 1
                elif e[j] == ')':
                    if depth == 0: break
                    depth -= 1
                j += 1
            inner = e[i+len("ceildiv("):j]
            # split on top-level comma
            cdepth = 0; comma = -1
            for k, ch in enumerate(inner):
                if ch == '(': cdepth += 1
                elif ch == ')': cdepth -= 1
                elif ch == ',' and cdepth == 0: comma = k; break
            if comma < 0: raise ValueError(f"ceildiv missing comma: {inner}")
            a, b = inner[:comma], inner[comma+1:]
            e = e[:i] + f"ceil(({a})/({b}))" + e[j+1:]
        # ... existing grammar continues unchanged ...
```

- [ ] **Step 2: 写 failing test**

新建 `test/python/test_network_runner_eval.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from network_runner import eval_block_dim

def test_ceildiv_basic():
    space = {"block_dim_expr": "ceildiv(s,8)"}
    assert eval_block_dim(space, {"s": 16}) == 2
    assert eval_block_dim(space, {"s": 17}) == 3

def test_ceildiv_nested():
    space = {"block_dim_expr": "(ceildiv(a*b,32)*32)"}
    # a=16, b=512 → a*b=8192 → ceildiv(8192,32)=256 → 256*32=8192
    assert eval_block_dim(space, {"a": 16, "b": 512}) == 8192

if __name__ == "__main__":
    test_ceildiv_basic(); test_ceildiv_nested()
    print("ok")
```

```bash
python3 test/python/test_network_runner_eval.py
```

Expected: `ok`(应该已经通过,因为 _eval 改在 Step 1 实施)。若先写测试再改:第一次 run 应 raise。

- [ ] **Step 3: picker 加 UB prune + PickerError**

在 `python/network_runner.py` 文件顶部加:

```python
class PickerError(RuntimeError):
    """Raised when no valid tiling candidate survives the budget filter."""
```

在 `phase3_default_build_and_dump` 第 ~308 行 `for p in space.get("tiling_params", []):` 循环改成:

```python
        ub_budget = int(space.get("ub_budget_bytes", 0))
        ub_cost_expr = space.get("ub_cost_bytes_expr") or ""
        chosen: dict = {}  # name -> picked value, built left-to-right
        for p in space.get("tiling_params", []):
            if not p.get("fixed", False):
                vals = p.get("values", []) or [16]
                if extent > 0:
                    capped = [v for v in vals if v <= extent]
                    vals = capped if capped else [extent]
                # UB-aware prune: evaluate ub_cost_expr with all known
                # fixed shape_keys + already-chosen tunables + this candidate.
                if ub_budget and ub_cost_expr:
                    survivors = []
                    for v in vals:
                        env = {**shape_keys, **chosen, p["name"]: v}
                        try:
                            cost = eval_block_dim({"block_dim_expr": ub_cost_expr}, env)
                        except Exception:
                            cost = 0  # unevaluable → don't prune
                        if not cost or cost <= ub_budget:
                            survivors.append(v)
                    if not survivors:
                        raise PickerError(
                            f"Kernel {vkid}: no tunable value of '{p['name']}' "
                            f"keeps UB cost ≤ {ub_budget} B (values={vals}, "
                            f"shapes={shape_keys}). Reduce extent, raise budget, "
                            f"or enable RBLOCK tile (see 2026-05-14-ub-aware-tiling-cost.zh.md).")
                    vals = survivors
                picked = vals[-1]
                chosen[p["name"]] = picked
                params[p["name"]] = picked
```

注:`chosen` 保留已选 tunable,让后续 tunable 的 cost 评估能用上下文(比如 XBLOCK 已选 → XBLOCK_SUB 评估时 XBLOCK 是已知量)。

- [ ] **Step 4: 测试 R=512 default 现在 PASS**

```bash
source /home/gser/Ascend/ascend-toolkit/set_env.sh >/dev/null
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
export PATH=$PWD/build/bin:$PATH
W=/tmp/dyn_t4 && rm -rf "$W" && mkdir -p "$W"
python3 examples/dyn-bucketed-e2e/gen_inputs.py --out-dir "$W" --d0 8 --d1 16 --d2 512 >/dev/null
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg examples/dyn-bucketed-e2e/model.mlir \
  --inputs $W/{a,b,c,d,e,f,g,init0,init1}.npy \
  --expected $W/expected0.npy $W/expected1.npy \
  --workdir $W --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 \
  --max-phase 3 2>&1 | tail -5
python3 -c "
import json, numpy as np
t = json.load(open('$W/tilings_default.json'))['kernel_group0__v0']
e = np.load('$W/expected0.npy'); g = np.load('$W/output_default_0.npy')
print('picked:', t['XBLOCK'], t['XBLOCK_SUB'], 'max_diff:', float(np.max(abs(e-g))))"
```

Expected:
- `XBLOCK` 仍 128
- `XBLOCK_SUB` 不再是 128,降到 16 或 32
- `max_diff` < 1e-5

- [ ] **Step 5: Commit**

```bash
git add python/network_runner.py test/python/test_network_runner_eval.py
git commit -m "feat(network-runner): UB-aware default-tile picker

Phase-3 default picker now evaluates the new ub_cost_bytes_expr
emitted by CannTranslation against the runtime shape, pruning
candidate values whose UB total exceeds ub_budget_bytes.  When all
candidates are filtered the picker fails fast with PickerError
pointing at the design doc.  Adds 'ceildiv(a,b)' to the eval grammar
to match SymExpr::emitC output.

Fixes dyn-bucketed-e2e all-zero output at d2 ≥ 256 (was silently
corrupting because XBLOCK_SUB=128 made each TBuf bump-pointer past
the 184KB UB pool).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: autotuner — UB-aware prune

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp:215-230` (TilingSpace struct)
- Modify: `tools/autotuner/autotuner_main.cpp:497-510` (loadSpace 读字段)
- Modify: `tools/autotuner/autotuner_main.cpp:690-720` (在 `bad =` 剪枝 之后加 UB cost prune)

- [ ] **Step 1: 扩展 TilingSpace**

在 `struct TilingSpace` 加字段:

```cpp
struct TilingSpace {
  std::string kernel_name;
  std::string kernel_file;
  std::string kernel_type = "vec";
  std::string soc;
  std::string block_dim_expr;
  std::string ub_cost_bytes_expr;  // new
  int64_t     ub_budget_bytes = 0; // new (0 = no budget known)
  std::vector<TilingParam> params;
};
```

- [ ] **Step 2: loadSpace 读字段**

在 space.json 解析处(约 497 行附近,处理完 `block_dim_expr` 之后)加:

```cpp
  if (auto s = po->getString("ub_cost_bytes_expr"))
    ts.ub_cost_bytes_expr = s->str();
  if (auto i = po->getInteger("ub_budget_bytes"))
    ts.ub_budget_bytes = *i;
```

- [ ] **Step 3: enumerate 后 prune**

在 enumerate 循环(约 695-710)`if (bad) { ++pruned; continue; }` 之前或之后加:

```cpp
    // UB-budget prune: if the kernel exposes ub_cost_bytes_expr, evaluate it
    // with `vars` (shape + tunables) and skip candidates that exceed the
    // budget.
    if (!ts.ub_cost_bytes_expr.empty() && ts.ub_budget_bytes > 0) {
      int64_t cost = evalBlockDimExpr(ts.ub_cost_bytes_expr, vars);
      if (cost > ts.ub_budget_bytes) { ++pruned; continue; }
    }
```

注:`evalBlockDimExpr` 是 C++ 端的 evaluator;需要确认它支持 `ceildiv(a,b)`。

- [ ] **Step 4: 检查 evalBlockDimExpr 语法支持**

```bash
grep -nE "ceildiv|ceil\(" tools/autotuner/autotuner_main.cpp | head
```

若不支持,在 `evalBlockDimExpr` 内加 `ceildiv(a,b)` → `(a+b-1)/b` 的预处理(类似 Python 端 Task 4 Step 1)。

- [ ] **Step 5: 构建 + 验证 prune 计数**

```bash
cmake --build build -j 8 2>&1 | tail -3
# 重跑 dyn-bucketed-e2e 直到 phase 4
W=/tmp/dyn_t5 && rm -rf "$W" && mkdir -p "$W"
python3 examples/dyn-bucketed-e2e/gen_inputs.py --out-dir "$W" --d0 8 --d1 16 --d2 512 >/dev/null
source /home/gser/Ascend/ascend-toolkit/set_env.sh >/dev/null
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
export PATH=$PWD/build/bin:$PATH
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg examples/dyn-bucketed-e2e/model.mlir \
  --inputs $W/{a,b,c,d,e,f,g,init0,init1}.npy \
  --expected $W/expected0.npy $W/expected1.npy \
  --workdir $W --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 \
  --max-phase 4 2>&1 | grep -E 'Pruned|FAIL|PASS|best' | head
```

Expected: "Pruned X of 16 tiling combos"(X 显著 > 6,因为加了 UB-prune)。最终能选出 best。

- [ ] **Step 6: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "feat(autotuner): UB-aware candidate prune

Read ub_cost_bytes_expr + ub_budget_bytes from tiling_space.json and
prune enumerate combos whose UB total exceeds budget.  Counts toward
the existing 'Pruned N of M' message.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: dyn-bucketed-e2e R sweep + regression + memory update

**Files:**
- Create: `examples/dyn-bucketed-e2e/run_R_sweep.sh`
- Modify: `examples/dyn-bucketed-e2e/BUG_REPORT.md`
- Modify: `/home/gser/.claude/projects/-home-gser-code-Ascend-MLIR/memory/MEMORY.md`
- Create: `/home/gser/.claude/projects/-home-gser-code-Ascend-MLIR/memory/project_ub_aware_tiling_cost.md`

- [ ] **Step 1: 写 sweep 脚本**

```bash
cat > examples/dyn-bucketed-e2e/run_R_sweep.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source /home/gser/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:/home/gser/Ascend/cann-9.0.0/x86_64-linux/lib64:/home/gser/Ascend/cann-9.0.0/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH:-}"
export PATH="$REPO/build/bin:$PATH"

for R in 64 128 256 512 1024; do
  W="/tmp/dyn_sweep_R$R"
  rm -rf "$W" && mkdir -p "$W"
  python3 "$SCRIPT_DIR/gen_inputs.py" --out-dir "$W" --d0 8 --d1 16 --d2 "$R" >/dev/null
  cd "$REPO"
  if PYTHONPATH=python python3 python/network_runner.py \
       --input-linalg "$SCRIPT_DIR/model.mlir" \
       --inputs "$W"/{a,b,c,d,e,f,g,init0,init1}.npy \
       --expected "$W"/expected0.npy "$W"/expected1.npy \
       --workdir "$W" --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 \
       2>&1 | grep -qE 'PASS$'; then
    echo "R=$R: PASS"
  else
    echo "R=$R: FAIL"
  fi
done
EOF
chmod +x examples/dyn-bucketed-e2e/run_R_sweep.sh
```

- [ ] **Step 2: 跑 sweep + 跑回归**

```bash
bash examples/dyn-bucketed-e2e/run_R_sweep.sh 2>&1 | tail
```

Expected: 至少 R=64..512 全 PASS。R=1024 可能 PASS 也可能因预算耗尽而 PickerError(后者也算预期,验证 fail-fast 报错链)。

回归(典型 9 个 e2e):
```bash
for ex in add-mul-relu-e2e bcast-leading-e2e bcast-middle-e2e bcast-trailing-e2e \
          bcast-multi-axis-e2e combo-elewise-reduce-e2e two-elewise-e2e \
          torch_e2e mixed-attn-e2e; do
  if [ -x "examples/$ex/run.sh" ]; then
    bash "examples/$ex/run.sh" >/dev/null 2>&1 && echo "$ex PASS" || echo "$ex FAIL"
  fi
done
```

Expected: 全 PASS。

- [ ] **Step 3: 更新 BUG_REPORT.md**

在 `examples/dyn-bucketed-e2e/BUG_REPORT.md` 文件头插一段(在现有 "Status: FIXED ..." 行之上):

```markdown
**2026-05-14 Update: R≥256 failure (different bug from §5)**

After commit `b3d196e` the d2=16 case passed, but d2 ≥ 256 still
produced all-zero output.  Root cause: the phase-3 default-tile
picker selected the **largest** XBLOCK_SUB candidate without regard
to UB capacity; at XBLOCK_SUB=128 + R=256 each TBuf init bump-pointer
allocated 128 KB on a 184 KB pool, and subsequent allocations
silently aliased past the end of UB.  Fixed by the
"UB-aware tiling cost" plan
(`docs/superpowers/plans/2026-05-14-ub-aware-tiling-cost.zh.md`):
CannTranslation now emits a symbolic `ub_cost_bytes_expr` derived
from the IR's `init_buffer/init_queue` size operands, plus an SoC
budget; pickers prune candidates that exceed the budget.
```

- [ ] **Step 4: 写 memory**

```bash
cat > /home/gser/.claude/projects/-home-gser-code-Ascend-MLIR/memory/project_ub_aware_tiling_cost.md <<'EOF'
---
name: ub-aware-tiling-cost
description: "CannTranslation emits ub_cost_bytes_expr (sum of init_buffer/init_queue sizes as SymExpr) + ub_budget_bytes (SoC TOTAL_VEC_LOCAL_SIZE); picker/autotuner prune candidates whose UB cost exceeds budget. Fixed dyn-bucketed-e2e R≥256 all-zero output."
metadata:
  type: project
---

**Status: shipped on llm-net 2026-05-14.**

Implements the Y branch of the UB-overrun investigation (see
[[multi-input-dyn-reduce-bug]] follow-up at d2 ≥ 256).

**Key parameters (CANN kernel_utils_constants.h):**
- 910B1 (NPU_ARCH=2201): TOTAL_UB_SIZE=192 KB, TOTAL_VEC_LOCAL_SIZE=184 KB.
- 910A (1001/2002): 256 KB / 248 KB.
- 310B (3003/3113): 118 KB / 118 KB.

**How the budget threshold was empirically validated:** at the
verified pre-fix HEAD (commit `2c97f95`), with the example
`examples/dyn-bucketed-e2e` driven from `gen_inputs.py --d0 8 --d1 16
--d2 R`, the picker chose XBLOCK=128 / XBLOCK_SUB=128, giving a
per-buffer size of `v32 = XBLOCK_SUB × R × sizeof(f32)`:

| R | v32 | observed |
|---|---|---|
| 192 | 96 KB | PASS (max_diff 2.9e-6) |
| 256 | 128 KB | FAIL all-zero (max_diff 28.5) |
| 512 | 256 KB | FAIL all-zero |
| 512 + XBLOCK_SUB=16 | 32 KB | PASS (max_diff 4.8e-6) |

So a single per-buffer size > ~100 KB starts crossing into "writes
silently fall off the end of UB" territory.  This is the bump-pointer
allocator in `TPipe::InitBuffer` (cann
`asc/impl/basic_api/kernel_tpipe_impl.h`) where assertions only log,
they don't abort, so overflow becomes a silent miscompile.

**How to apply:** if a new symptom of "kernel ran, no crash, all-zero
or partially-corrupt output" appears on dynamic-shape reduce-like
kernels, first check `<kid>_space.json` for ub_cost_bytes_expr and
ub_budget_bytes — if either is missing the picker can't prune.
Cross-check with `eval_block_dim` from network_runner under the
runtime shape to see the actual cost.  When all candidates are
pruned the picker raises PickerError pointing at this plan — that's
the cue to enable RBLOCK splitting (separate plan, not yet written).

Related:
- [[multi-input-dyn-reduce-bug]] — sibling bug at d2=16, fixed by
  symbolic-shape propagation through alloc_tensor copy.
- [[reduce-path-r3]] R1 partial+combine — alternative path for very
  large R (would still be needed if budget exhausts).
- [[af_scheduler_port]] P6 cost-model TODO — this plan completes a
  slice of P6 (UB-byte cost only; cycle cost still missing).
EOF
```

- [ ] **Step 5: 更新 MEMORY.md 索引**

在 `/home/gser/.claude/projects/-home-gser-code-Ascend-MLIR/memory/MEMORY.md` 中追加一行:

```markdown
- [UB-aware tiling cost](project_ub_aware_tiling_cost.md) — CannTranslation emits ub_cost_bytes_expr + ub_budget_bytes; picker/autotuner prune over-budget candidates. Fixed dyn-bucketed-e2e R≥256.
```

- [ ] **Step 6: Commit**

```bash
git add examples/dyn-bucketed-e2e/run_R_sweep.sh \
        examples/dyn-bucketed-e2e/BUG_REPORT.md
git commit -m "test(dyn-bucketed-e2e): R sweep + mark UB-aware fix in BUG_REPORT

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

memory 文件不进 git。

---

## Self-Review checklist

- [ ] Task 1-6 都有 commit 步骤
- [ ] Task 3 测试 `test/CannTranslation/ub-cost-expr.mlir` 在改 emit 前会 FAIL,改后 PASS — TDD 链完整
- [ ] Task 4 加 `PickerError`,候选全空时报错而不是回退到 buggy 行为 — fail-fast
- [ ] `--soc` 参数 threading 还是 TODO(Task 3 Step 2 注释里标了);若 plan 范围内必须修,挪到 Task 3 Step 0
- [ ] SymExpr `emitC` 输出语法(`ceildiv(a,b)`)与 Python `_eval` / C++ `evalBlockDimExpr` 兼容由 Task 4 Step 1 / Task 5 Step 4 保证
- [ ] 实测阈值(v32=96KB PASS、128KB FAIL)与公式 `184 KB / 单 buffer 上限` 一致;若 picker 公式过于保守(R=192 时把 128 也剪掉),不影响正确性,但可能影响性能 — 留作后续 follow-up,不阻塞本 plan
- [ ] 不涉及 RBLOCK 切分;不修 autotuner false-pass;两者各自独立 follow-up
