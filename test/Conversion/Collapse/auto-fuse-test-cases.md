# AutoFuse 测试用例开发说明

## 目标

`auto-fuse-group-analysis` 和 `auto-fuse-group-outline` 是严格串联的两个 pass：前者只负责给 linalg op 打 `auto_fuse.*` 属性，后者消费这些属性并把 group outline 成 kernel func。

因此测试优先采用联测，而不是把两个模块完全拆开。联测更能验证真实流水线行为：

1. `GroupAnalysis` 是否把 op 分到正确的 group。
2. `GroupOutline` 是否按这些 group 生成 kernel func。
3. coordinator 是否正确改写为 `func.call`。
4. 输出中是否已经清除 `auto_fuse.*` 属性。

## 当前新增的联测

### 1. `group-pipeline-single.mlir`

覆盖最小闭环：单个 linalg op 被 outline 为一个 kernel func。

重点检查：
- coordinator 中只有一次 `func.call`
- kernel func 中保留原始 `linalg.generic`
- 输出中不含 `auto_fuse.group_id/topo_index`

### 2. `group-pipeline-vertical.mlir`

覆盖 vertical fusion：`reduce -> pointwise` 应先被分析为同一 group，再 outline 为一个 kernel func。

重点检查：
- coordinator 只调用一个 kernel func
- kernel body 中包含两个 `linalg.generic`
- 最终返回值来自该 call

### 3. `group-pipeline-horizontal.mlir`

覆盖 horizontal fusion：两个 sibling op 共享同一个 boundary input，应进入同一 group。

重点检查：
- coordinator 对应一次多结果 `func.call`
- kernel func 返回两个 tensor
- kernel body 中包含两个 sibling `linalg.generic`

### 4. `group-pipeline-no-fuse-fanout.mlir`

覆盖 no-fuse 场景：中间结果被多个 group 外消费者使用时，不应发生 vertical fusion。

重点检查：
- pipeline 最终产生两个 kernel func
- 第一个 kernel 产出中间值
- 第二个 kernel 同时消费中间值并返回两个结果

## 断言策略

为减少对实现细节的脆弱依赖，测试遵循以下规则：

- 统一使用 `afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline`
- 用最终 IR 结构间接验证 group 是否正确，而不是依赖中间 debug 输出
- `kernel_group` 编号通过 FileCheck capture 匹配，不写死具体数值
- 不测试当前尚未实现的功能，例如：
  - `kernelFuncPrefix`
  - `outputDir`
  - TileFuse / TileInfo
  - cube + vector 的完整后续流水线

## 后续建议补充的用例

后续如果继续扩展，可以优先补这些联测：

1. 多轮融合场景（例如 layernorm / skip connection）
2. 多 group 顺序验证
3. 多结果 boundaryOut 更复杂的场景
4. 多 func module 场景
5. cube epilogue / prologue 场景（等实现打开后再测）

## 运行方式

### 1. 环境准备

如果当前 shell 里还没有 `afir-opt`，先加载项目环境：

```bash
export ASCEND_HOME_PATH=/home/gser/Ascend
source examples/env.sh
```

检查工具是否已经就绪：

```bash
command -v afir-opt
```

如果本机有 LLVM/MLIR 的测试工具，也可以顺手检查：

```bash
command -v FileCheck
```

### 2. 运行单个测试文件

只看 pass 输出：

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-vertical.mlir
```

如果环境里有 `FileCheck`，可以直接按测试文件内的 `CHECK` 做校验：

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-vertical.mlir \
  | FileCheck test/Conversion/AutoFuse/group-pipeline-vertical.mlir
```

### 3. 逐个运行当前联测

#### `group-pipeline-single.mlir`

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-single.mlir
```

#### `group-pipeline-vertical.mlir`

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-vertical.mlir
```

#### `group-pipeline-horizontal.mlir`

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-horizontal.mlir
```

#### `group-pipeline-no-fuse-fanout.mlir`

```bash
afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline \
  test/Conversion/AutoFuse/group-pipeline-no-fuse-fanout.mlir
```

### 4. 只看 GroupAnalysis 标记后的 IR

如果想单独观察 `auto_fuse.group_id` / `auto_fuse.topo_index`，只跑 analysis pass：

```bash
afir-opt --auto-fuse-group-analysis \
  test/Conversion/AutoFuse/group-analysis-vertical.mlir
```

这个命令适合确认：
- 哪些 op 被分到同一个 group
- `topo_index` 是否符合预期顺序

### 5. 批量运行

如果只是想快速把 4 个联测都跑一遍，可以用一个简单循环：

```bash
for f in \
  test/Conversion/AutoFuse/group-pipeline-single.mlir \
  test/Conversion/AutoFuse/group-pipeline-vertical.mlir \
  test/Conversion/AutoFuse/group-pipeline-horizontal.mlir \
  test/Conversion/AutoFuse/group-pipeline-no-fuse-fanout.mlir; do
  echo "===== $f ====="
  afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline "$f"
done
```

如果环境里有 `FileCheck`，可以改成：

```bash
for f in test/Conversion/AutoFuse/group-pipeline-*.mlir; do
  echo "===== $f ====="
  afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline "$f" \
    | FileCheck "$f"
done
```

## 调试建议

单独调试某个测试时，先直接看 pass 输出，再决定是否加 `FileCheck`。

重点看：
- coordinator 里 call 的数量和顺序
- `kernel_group*` 的参数和返回类型
- kernel body 里的 linalg op 数量
- 是否已经移除了 `auto_fuse.*` 属性
