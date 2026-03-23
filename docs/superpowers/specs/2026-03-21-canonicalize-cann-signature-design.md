# CanonicalizeCannSignaturePass + cann-translate Design

**Date:** 2026-03-21
**Goal:** Make the MLIR pipeline output kernel.cpp that is directly compatible with the Compiler/Executor/Validator/Autotuner toolchain by conforming to the CANN standard kernel calling convention.

---

## Problem Statement

The current pipeline (`ascir-translate -mlir-to-ascendc`) generates kernel signatures in PyAsc's internal format:

```cpp
extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    half* input_a,
    half* input_b,
    __gm__ TilingData* tiling_data_ptr,   // tiling in the middle, pointer
    half* output
)
```

The CANN standard calling convention required by `bisheng` / the toolchain is:

```cpp
extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    GM_ADDR input_a,        // inputs first
    GM_ADDR input_b,
    GM_ADDR output,         // then outputs
    GM_ADDR workspace,      // then workspace (unused but required)
    TilingData tiling       // tiling last, by-value (not pointer)
)
```

Key differences:
1. Parameter order: `(inputs, tiling_ptr, outputs)` → `(inputs, outputs, workspace, tiling)`
2. GM pointer type: `__gm__ T*` → `GM_ADDR` (`uint8_t*`)
3. Tiling: pointer + byte-copy loop → by-value struct parameter
4. Extra `workspace` parameter inserted

---

## Solution: Two-Component Approach

### Component 1: `CanonicalizeCannSignaturePass` (MLIR Pass)

Transforms the `func.func` IR in-place to the CANN parameter layout.

**Input** (step7_kernel.mlir func signature):
```mlir
func.func @broadcast_add_reducesum(
    %input_a: memref<?xf16>,
    %input_b: memref<?x?xf16>,
    %tiling_data: memref<?x!emitasc.py_struct<"TilingData", [i64,i64,i64,i64],
                                               ["TB_M","TB_N","dim_arg0_0","dim_arg1_1"]>, 22:i32>,
    %output: memref<?xf16, strided<[1], offset: ?>>
) attributes {ascendc.aicore, ascendc.global}
```

**Output** (step7_cann.mlir func signature):
```mlir
func.func @broadcast_add_reducesum(
    %input_a: memref<?xf16>,
    %input_b: memref<?x?xf16>,
    %output: memref<?xf16, strided<[1], offset: ?>>,
    %workspace: memref<ui8>,
    %tiling: !emitasc.py_struct<"TilingData", [i64,i64,i64,i64],
                                 ["TB_M","TB_N","dim_arg0_0","dim_arg1_1"]>
) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32}
```

**Transformation steps:**

1. **Identify the tiling parameter**: the unique `memref<?x!emitasc.py_struct<...>, 22:i32>` argument. Verify only one exists; emit error otherwise.

2. **Determine num_inputs**: count of args before the tiling param that are NOT `memref<PyStruct>`. Record as `cann.num_inputs` function attribute.

3. **Rebuild argument list**: `[inputs..., outputs..., memref<ui8>, PyStructType]`
   - inputs = args[0..tiling_idx-1] (the non-tiling, non-output args before tiling)
   - outputs = args after tiling (currently exactly one output; assert count)
   - workspace = new `memref<ui8>` argument
   - tiling = new `!emitasc.py_struct<...>` argument (bare struct, not memref-wrapped)

4. **Rewrite function signature** using `mlir::FunctionType` replacement + `replaceAllArgUsesWith` / `insertArgument` / `eraseArgument`.

5. **Rewrite body**:
   - Remove the `emitasc.copy_struct %old_tiling_data` op.
   - Replace all uses of its result `%local_tiling` with the new `%tiling` parameter.
   - The `emitasc.member %tiling "FIELD"` ops remain unchanged — they already handle by-value `PyStructType` via the `.` path in PyAsc's `MemberOp` printer (`isa<MemRefType>` check → else branch uses `.`).

6. **Add function attribute**: `cann.num_inputs = N : i32` on the func.

**Pass registration** in `include/Conversion/Passes.td`:
```tablegen
def CanonicalizeCannSignaturePass : Pass<"canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize kernel function signature to CANN calling convention";
  let description = [{
    Transforms aicore kernel functions from PyAsc internal signature format
    (inputs, tiling_ptr, outputs) to CANN standard format
    (inputs, outputs, workspace, tiling_byvalue).
  }];
}
```

---

### Component 2: `cann-translate` (New Translation Registration)

Registered as `"mlir-to-cann"` in `afir-opt`. Handles the CANN-specific function signature emission while reusing PyAsc's `emitOperation()` for all op bodies.

**File:** `lib/Target/CannKernel/CannTranslation.cpp`

**Top-level entry:**
```cpp
LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os) {
    CodeEmitter emitter(os);
    auto moduleOp = dyn_cast<ModuleOp>(op);
    if (!moduleOp)
        return op->emitOpError("expected ModuleOp");
    CodeEmitter::Scope scope(emitter);
    for (Operation &child : *moduleOp.getBody()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(child)) {
            if (funcOp->hasAttr(ascendc::attr::global)) {
                if (failed(printCannFuncOp(emitter, funcOp)))
                    return failure();
                emitter.ostream() << "\n";
                continue;
            }
        }
        // Non-aicore funcs and all other ops: use PyAsc emitter as-is
        if (failed(emitOperation(emitter, child, /*trailingSemicolon=*/false)))
            return failure();
    }
    return success();
}
```

**`printCannFuncOp` signature emission:**

Reads `cann.num_inputs` from the func attribute to know how many leading args are inputs. Remaining non-workspace/non-tiling args are outputs.

```cpp
LogicalResult printCannFuncOp(CodeEmitter &emitter, func::FuncOp funcOp) {
    auto &os = emitter.ostream();
    auto args = funcOp.getArguments();
    int numInputs = cast<IntegerAttr>(funcOp->getAttr("cann.num_inputs")).getInt();

    // Partition: inputs | outputs | workspace | tiling
    // By convention (enforced by Pass):
    //   args[0..numInputs-1]      = inputs
    //   args[numInputs..N-3]      = outputs  (usually 1)
    //   args[N-2]                 = workspace (memref<ui8>)
    //   args[N-1]                 = tiling   (!emitasc.py_struct<...>)
    int N = args.size();
    auto tilingArg    = args[N-1];   // PyStructType
    auto workspaceArg = args[N-2];   // memref<ui8>

    // 1. Emit TilingData struct declaration
    auto pyStructType = cast<emitasc::PyStructType>(tilingArg.getType());
    emitTilingStructDecl(os, pyStructType);

    // 2. Emit function signature
    os << "extern \"C\" __global__ __aicore__ void "
       << funcOp.getName() << "(\n";
    os.indent();

    // inputs: GM_ADDR name
    for (int i = 0; i < numInputs; ++i) {
        os << "GM_ADDR " << emitter.getOrCreateName(args[i]) << ",\n";
    }
    // outputs: GM_ADDR name
    for (int i = numInputs; i < N-2; ++i) {
        os << "GM_ADDR " << emitter.getOrCreateName(args[i]) << ",\n";
    }
    // workspace
    os << "GM_ADDR " << emitter.getOrCreateName(workspaceArg) << ",\n";
    // tiling by-value
    os << pyStructType.getNameAttr().getValue()
       << " " << emitter.getOrCreateName(tilingArg) << "\n";

    os.unindent() << ") {\n";
    os.indent();

    // 3. Emit function body ops (reuse PyAsc emitOperation entirely)
    for (Operation &op : funcOp.getBody().front()) {
        if (failed(emitOperation(emitter, op, needsSemicolon(op))))
            return failure();
    }

    os.unindent() << "}\n";
    return success();
}
```

**`emitTilingStructDecl`** emits:
```cpp
struct TilingData {
    int64_t TB_M;
    int64_t TB_N;
    int64_t dim_arg0_0;
    int64_t dim_arg1_1;
};
```
Derived from `PyStructType::getTypesAttr()` and `PyStructType::getNamesAttr()`.

**Registration in `tools/afir-opt/afir-opt.cpp`:**
```cpp
#include "Target/CannKernel/CannTranslation.h"

// In main():
mlir::TranslateFromMLIRRegistration cannReg(
    "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
    [](Operation *op, raw_ostream &os) {
        return mlir::translateToCannKernel(op, os);
    },
    [](DialectRegistry &registry) {
        // Same dialect registrations as mlir-to-ascendc
        registry.insert<func::FuncDialect, emitasc::EmitAscDialect,
                        ascendc::AscendCDialect, ...>();
    });
```

---

## Output Example

**Input** (`step7_cann.mlir` after pass):
```mlir
func.func @broadcast_add_reducesum(
    %a: memref<?xf16>, %b: memref<?x?xf16>,
    %out: memref<?xf16, strided<[1],offset:?>>,
    %ws: memref<ui8>,
    %tiling: !emitasc.py_struct<"TilingData",[i64,i64,i64,i64],
                                 ["TB_M","TB_N","dim_arg0_0","dim_arg1_1"]>
) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32}
```

**Output** (`step8_kernel.cpp`):
```cpp
#include "kernel_operator.h"

struct TilingData {
    int64_t TB_M;
    int64_t TB_N;
    int64_t dim_arg0_0;
    int64_t dim_arg1_1;
};

extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    GM_ADDR v1,
    GM_ADDR v2,
    GM_ADDR v3,
    GM_ADDR v4,
    TilingData v5
) {
    int64_t v6 = v5.TB_M;
    int64_t v7 = v5.TB_N;
    // ... rest of body unchanged, emitted by PyAsc emitOperation() ...
}
```

---

## File Map

| File | Purpose |
|------|---------|
| `include/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h` | Pass header, declares `createCanonicalizeCannSignaturePass()` |
| `lib/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.cpp` | Pass implementation |
| `lib/Conversion/CanonicalizeCannSignature/CMakeLists.txt` | CMake target |
| `include/Target/CannKernel/CannTranslation.h` | Declares `translateToCannKernel()` |
| `lib/Target/CannKernel/CannTranslation.cpp` | Translation implementation |
| `lib/Target/CannKernel/CMakeLists.txt` | CMake target |
| `include/Conversion/Passes.td` | Add `CanonicalizeCannSignaturePass` entry |
| `include/Conversion/Passes.h` | Include new pass header |
| `lib/Conversion/CMakeLists.txt` | Add subdirectory |
| `tools/afir-opt/afir-opt.cpp` | Register `mlir-to-cann` translation |
| `tools/afir-opt/CMakeLists.txt` | Link new targets |
| `test/Conversion/canonicalize-cann-signature.mlir` | Lit test for the pass |
| `test/Target/cann-translate.mlir` | Lit test for the translator |
| `examples/broadcast-add-reduce/run.sh` | Add steps 7b + 8 using new tools |

---

## Run.sh Integration (broadcast-add-reduce)

```bash
# Stage 7b: canonicalize to CANN signature
afir-opt --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" -o "$DIR/step7_cann.mlir"

# Stage 8: emit CANN-standard kernel.cpp
afir-opt -mlir-to-cann "$DIR/step7_cann.mlir" -o "$DIR/step8_kernel.cpp"

# Stage 9: compile + generate runner
compiler \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$DIR/build" \
  --num-inputs 2 --num-outputs 1 \
  --tiling-layout "int64,int64,int64,int64" \
  --kernel-type vec
```

---

## Constraints and Assumptions

- Only `func.func` ops with `ascendc.aicore` + `ascendc.global` attributes are transformed.
- Exactly one tiling parameter (identified by `memref<?x!emitasc.py_struct<...>, 22:i32>`) per kernel function; pass emits an error if not found.
- `num_outputs` is derived implicitly as `N - numInputs - 2` in the CANN-form signature (workspace at `args[N-2]`, tiling at `args[N-1]`).
- `emitasc.copy_struct` must be the first use of the tiling memref arg and its result must only be used by `emitasc.member` ops; pass verifies this and emits an error otherwise.
- The `emitasc.member` op on a by-value `PyStructType` already emits `.` (not `->`) in PyAsc — no PyAsc changes needed.
- `needsSemicolon()` and `emitOperation()` are both declared in `Common.h` (not anonymous namespace); `CannTranslation.cpp` includes `Common.h` directly — no duplication needed.
- `emitTilingStructDecl` must use `CodeEmitter::emitType()` per field to map MLIR types (e.g., `i64` → `int64_t`), not hardcode type strings.
