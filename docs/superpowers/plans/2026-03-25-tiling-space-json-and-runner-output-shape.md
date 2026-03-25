# Tiling Space JSON Auto-generation & Runner Output Shape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) `afir-translate -mlir-to-cann` 生成 `.cpp` 时同步输出 `tiling_space.json` 骨架；(2) 生成的 `runner` 可执行文件支持 `--output-shape` 参数。

**Architecture:**
- 变更1：在 `translateToCannKernel()` 中，从 aicore func 的最后一个参数（`PyStructType`）提取字段名和类型，生成 `tiling_space.json` 骨架并写到 `--tiling-space-out` 指定的路径。字段分两类：`dim_arg*` → `fixed:true, shape_key:argN_dimD`；其余 → `fixed:false, values:[]`。使用 `llvm::json::OStream` 做 pretty-print。
- 变更2：`HostRunnerGen::emitRunnerCpp()` 生成的模板代码中加入 `--output-shape` 参数解析，优先使用它，fallback 到 `inputs[0].shape`。注意编辑的是 `HostRunnerGen.cpp` 内的 raw string 模板。

**Tech Stack:** C++17, LLVM/MLIR, llvm::json, llvm::CommandLine

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `include/Target/CannKernel/CannTranslation.h` | Modify | `translateToCannKernel` 增加 `tilingSpaceOutPath` 参数 |
| `lib/Target/CannKernel/CannTranslation.cpp` | Modify | 实现 JSON 骨架生成逻辑 |
| `tools/afir-translate/afir-translate.cpp` | Modify | 注册 `--tiling-space-out` CLI 参数 |
| `lib/Runtime/HostRunnerGen.cpp` | Modify | 模板代码（raw string 内）加入 `--output-shape` 解析 |

---

## Task 1: `translateToCannKernel` 增加 tiling_space.json 骨架输出

**Files:**
- Modify: `include/Target/CannKernel/CannTranslation.h`
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Modify: `tools/afir-translate/afir-translate.cpp`

### 背景

`translateToCannKernel()` 第一遍遍历 aicore funcs 时，已经拿到每个 func 最后一个参数的 `emitasc::PyStructType`，其中包含所有字段名和类型。此时可以同步生成 JSON 骨架。

字段命名规则：
- `dim_argN_D` 格式 → `"fixed": true, "shape_key": "argN_dimD"`
- 其他（TB_M 等）→ `"fixed": false, "values": []`

JSON 结构：
```json
{
  "kernel": "<funcOp name>",
  "kernel_file": "",
  "soc": "Ascend910B1",
  "block_dim_expr": "",
  "tiling_params": [...]
}
```

- [ ] **Step 1: 修改头文件签名**

`include/Target/CannKernel/CannTranslation.h`，替换现有声明：

```cpp
/// Translates a module containing CANN-signature aicore functions to C++.
/// Expects func.func args in order: inputs, outputs, workspace:memref<ui8>,
/// tiling:!emitasc.py_struct<...>, with cann.num_inputs attr.
/// tilingSpaceOutPath: if non-empty, write tiling_space.json skeleton to this path.
/// kernelFile: value for "kernel_file" field in the JSON (may be empty).
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    StringRef tilingSpaceOutPath = "",
                                    StringRef kernelFile = "");
```

- [ ] **Step 2: 实现 JSON 生成辅助函数**

注：`llvm/Support/JSON.h` 已由现有的 `LLVMSupport` 依赖提供，无需修改 CMakeLists.txt。

在 `lib/Target/CannKernel/CannTranslation.cpp` 顶部已有 includes 后追加：

```cpp
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
```

在 `emitTilingStructDecl` 函数之后、`printCannFuncOp` 之前插入：

```cpp
/// Write tiling_space.json skeleton to outPath.
/// dim_argN_D fields → fixed:true, shape_key:"argN_dimD".
/// Other fields (TB_M etc.) → fixed:false, values:[].
static void emitTilingSpaceJson(StringRef outPath,
                                StringRef kernelFile,
                                StringRef kernelName,
                                emitasc::PyStructType tilingType) {
  auto isDimField = [](StringRef name) {
    return name.starts_with("dim_arg");
  };
  // "dim_arg2_1" → drop "dim_" → "arg2_1" → rfind '_' → "arg2" + "_dim" + "1" → "arg2_dim1"
  auto makeShapeKey = [](StringRef name) -> std::string {
    StringRef rest = name.drop_front(4); // drop "dim_"
    auto pos = rest.rfind('_');
    if (pos == StringRef::npos) return rest.str();
    return rest.substr(0, pos).str() + "_dim" + rest.substr(pos + 1).str();
  };

  auto names = tilingType.getNamesAttr().getValue();

  llvm::json::Array params;
  for (auto& nameAttr : names) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    llvm::json::Object p;
    p["name"] = name.str();
    p["type"] = "int64";
    if (isDimField(name)) {
      p["fixed"] = true;
      p["shape_key"] = makeShapeKey(name);
    } else {
      p["fixed"] = false;
      p["values"] = llvm::json::Array{};
    }
    params.push_back(std::move(p));
  }

  llvm::json::Object root;
  root["kernel"]        = kernelName.str();
  root["kernel_file"]   = kernelFile.str();
  root["soc"]           = "Ascend910B1";
  root["block_dim_expr"]= "";
  root["tiling_params"] = std::move(params);

  std::error_code ec;
  llvm::raw_fd_ostream f(outPath, ec);
  if (ec) {
    llvm::errs() << "Warning: cannot write tiling_space.json to "
                 << outPath << ": " << ec.message() << "\n";
    return;
  }
  // Use llvm::json::OStream for pretty-print (formatv does not support json::Value)
  llvm::json::OStream jos(f, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  f << "\n";
}
```

- [ ] **Step 3: 修改 `translateToCannKernel` 函数签名和第一遍遍历**

`lib/Target/CannKernel/CannTranslation.cpp`，修改函数定义头：

```cpp
LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os,
                                          StringRef tilingSpaceOutPath,
                                          StringRef kernelFile) {
```

在第一遍遍历中，`emitTilingStructDecl` 调用之后加 JSON 输出（用 `bool jsonWritten` 保证只写一次）：

```cpp
  // First pass: emit TilingData struct declarations from aicore funcs
  bool jsonWritten = false;
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp)
      continue;
    if (!funcOp->hasAttr(ascendc::attr::global))
      continue;

    auto args = funcOp.getArguments();
    if (args.empty())
      continue;
    auto tilingType =
        dyn_cast<emitasc::PyStructType>(args.back().getType());
    if (!tilingType)
      continue;

    if (failed(emitTilingStructDecl(emitter, funcOp.getLoc(), tilingType)))
      return failure();

    // Write JSON skeleton for the first aicore func only
    if (!tilingSpaceOutPath.empty() && !jsonWritten) {
      emitTilingSpaceJson(tilingSpaceOutPath, kernelFile,
                          funcOp.getName(), tilingType);
      jsonWritten = true;
    }
  }
```

- [ ] **Step 4: 在 `afir-translate.cpp` 注册 CLI 参数**

在 `main()` 之前加（`#include "llvm/Support/CommandLine.h"` 已有则不重复）：

```cpp
static llvm::cl::opt<std::string> TilingSpaceOut(
    "tiling-space-out",
    llvm::cl::desc("Write tiling_space.json skeleton to this path"),
    llvm::cl::init(""));
```

修改 lambda（`kernelFile` 留空，后续可从 `--output` 推断）：

```cpp
    [](Operation *op, raw_ostream &os) {
      return translateToCannKernel(op, os, TilingSpaceOut, "");
    },
```

- [ ] **Step 5: 编译验证**

在 VM 中：
```bash
cd /home/niu/code/Ascend-MLIR
./scripts/build.sh --build-project
```
期望：编译无错误。

- [ ] **Step 6: 手动测试**

```bash
source examples/env.sh
afir-translate -mlir-to-cann \
  examples/broadcast-add-reduce/step7_cann.mlir \
  -o /tmp/test_kernel.cpp \
  --tiling-space-out /tmp/test_tiling_space.json
cat /tmp/test_tiling_space.json
```

期望输出（字段顺序可能因 JSON 库实现而异）：
```json
{
  "kernel": "broadcast_add_reducesum",
  "kernel_file": "",
  "soc": "Ascend910B1",
  "block_dim_expr": "",
  "tiling_params": [
    {"fixed": false, "name": "TB_M", "type": "int64", "values": []},
    {"fixed": false, "name": "TB_N", "type": "int64", "values": []},
    {"fixed": true, "name": "dim_arg0_0", "shape_key": "arg0_dim0", "type": "int64"},
    {"fixed": true, "name": "dim_arg1_1", "shape_key": "arg1_dim1", "type": "int64"},
    {"fixed": true, "name": "dim_arg0_1", "shape_key": "arg0_dim1", "type": "int64"},
    {"fixed": true, "name": "dim_arg1_0", "shape_key": "arg1_dim0", "type": "int64"}
  ]
}
```

- [ ] **Step 7: Commit**

```bash
git add include/Target/CannKernel/CannTranslation.h \
        lib/Target/CannKernel/CannTranslation.cpp \
        tools/afir-translate/afir-translate.cpp
git commit -m "feat(translate): add --tiling-space-out to auto-generate tiling_space.json skeleton"
```

---

## Task 2: runner 支持 `--output-shape` 参数

**Files:**
- Modify: `lib/Runtime/HostRunnerGen.cpp`

### 背景

生成的 `runner` 可执行文件当前硬编码 `output.shape = inputs[0].shape`（`emitRunnerCpp()` 内 raw string 模板，行 ~214-218）。需要加 `--output-shape 64,64` 参数，不指定时 fallback 到 `inputs[0].shape`。

**重要**：编辑目标是 `lib/Runtime/HostRunnerGen.cpp` 中 `emitRunnerCpp()` 函数内的 `R"cpp(...)cpp"` raw string 字面量，而不是任何生成文件。

- [ ] **Step 1: 在变量声明段加 `output_shape_str`**

定位 raw string 内的变量声明行（当前内容）：
```cpp
  std::string bin_path, tiling_params, tiling_layout_str, inputs_str,
              output_path = "/dev/null";
```

改为：
```cpp
  std::string bin_path, tiling_params, tiling_layout_str, inputs_str,
              output_path = "/dev/null", output_shape_str;
```

- [ ] **Step 2: 在 arg 解析段加 `--output-shape` 分支**

定位 raw string 内的 arg 解析段，在 `--output` 行之后、`--block-dim` 行之前加一行：

```cpp
    else if (a == "--output-shape")  output_shape_str  = next();
```

完整段落变为：
```cpp
    if      (a == "--bin")           bin_path          = next();
    else if (a == "--tiling-params") tiling_params     = next();
    else if (a == "--tiling-layout") tiling_layout_str = next();
    else if (a == "--inputs")        inputs_str        = next();
    else if (a == "--output")        output_path       = next();
    else if (a == "--output-shape")  output_shape_str  = next();
    else if (a == "--block-dim")     block_dim         = std::stoi(next());
```

- [ ] **Step 3: 替换 output 分配段**

定位 raw string 内（行 ~214-218）：
```cpp
  // Allocate output (single output; same shape/dtype as inputs[0])
  NDArray output;
  output.shape = inputs[0].shape;
  output.dtype = inputs[0].dtype;
  output.data  = new uint8_t[output.nbytes()]();
```

改为：
```cpp
  // Allocate output (single output)
  NDArray output;
  if (!output_shape_str.empty()) {
    // Parse --output-shape "64,64" → {64, 64}
    auto shape_parts = splitComma(output_shape_str);
    for (auto& s : shape_parts)
      output.shape.push_back(std::stoll(s));
  } else {
    output.shape = inputs[0].shape;
  }
  output.dtype = inputs[0].dtype;
  output.data  = new uint8_t[output.nbytes()]();
```

- [ ] **Step 4: 编译验证**

```bash
cd /home/niu/code/Ascend-MLIR
./scripts/build.sh --build-project
```
期望：编译无错误。

- [ ] **Step 5: 手动测试**

```bash
source examples/env.sh

# 先编译 kernel（同时生成 runner）
compiler \
  --kernel examples/broadcast-add-reduce/step8_kernel.cpp \
  --output /tmp/test_build \
  --name broadcast_add_reducesum \
  --num-inputs 2

# 不带 --output-shape（fallback）
/tmp/test_build/runner \
  --bin /tmp/test_build/broadcast_add_reducesum.bin \
  --inputs examples/broadcast-add-reduce/input_a.npy,examples/broadcast-add-reduce/input_b.npy \
  --output /tmp/out_fallback.npy \
  --tiling-params '16,16,64,64,64,64' \
  --block-dim 4

# 带 --output-shape（broadcast-add-reduce 输出是 reducesum 结果，shape=64,1 或 64）
# 具体 shape 取决于 kernel 实现，先用 fallback shape 确认后再调整
/tmp/test_build/runner \
  --bin /tmp/test_build/broadcast_add_reducesum.bin \
  --inputs examples/broadcast-add-reduce/input_a.npy,examples/broadcast-add-reduce/input_b.npy \
  --output /tmp/out_explicit.npy \
  --output-shape 64,64 \
  --tiling-params '16,16,64,64,64,64' \
  --block-dim 4
```

期望：两次运行均正常完成，`/tmp/out_fallback.npy` 和 `/tmp/out_explicit.npy` 生成成功。具体数值与 `examples/broadcast-add-reduce/output_c.npy` 对比以确认 `--output-shape` 值。

- [ ] **Step 6: Commit**

```bash
git add lib/Runtime/HostRunnerGen.cpp
git commit -m "feat(runner): add --output-shape parameter to generated runner executable"
```

---

## Task 3: 更新 run.sh 使用 --tiling-space-out

**Files:**
- Modify: `examples/broadcast-add-reduce/run.sh`

- [ ] **Step 1: 在 STAGE 8（afir-translate）加 `--tiling-space-out`**

定位 run.sh 中的 afir-translate 调用（当前）：
```bash
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" -o "$DIR/step8_kernel.cpp" 2>&1
```

改为：
```bash
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" \
  --tiling-space-out "$DIR/step8_kernel.tiling_space.json" 2>&1
```

- [ ] **Step 2: 运行 run.sh 并验证**

```bash
cd examples/broadcast-add-reduce
source ../../examples/env.sh
bash run.sh
cat step8_kernel.tiling_space.json
```

期望：`step8_kernel.tiling_space.json` 存在且内容正确。

- [ ] **Step 3: Commit**

```bash
git add examples/broadcast-add-reduce/run.sh
git commit -m "feat(example): emit tiling_space.json skeleton in run.sh codegen stage"
```

---

## 注意事项

1. `llvm::json::OStream` 用于 pretty-print，`llvm::formatv` 不支持 `json::Value`。
2. Task 2 的所有编辑都在 `HostRunnerGen.cpp` 的 `emitRunnerCpp()` 函数内的 `R"cpp(...)cpp"` raw string 字面量中，不要编辑任何生成的文件。
3. `bool jsonWritten` 保证多 kernel module 只生成一份 JSON。
