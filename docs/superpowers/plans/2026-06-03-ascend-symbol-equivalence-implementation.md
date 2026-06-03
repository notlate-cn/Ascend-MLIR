# Ascend Symbol Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the V2 Layer 1 symbolic dimension equivalence contract by producing and verifying `ascend.symbol_constraints` from Normalize.

**Architecture:** Add a shared symbol-constraint helper layer that owns attr names, function-local value ordinals, serialization, parsing, and lookup. Add a Normalize-only symbol equivalence analysis that uses internal `DimRef(Value, dim)` nodes and a weighted Union-Find to apply R1-R6, then attach the serialized attr to each `func.func`. Add an entry verifier that validates the attr structurally and recomputes proof edges to fail closed when Normalize output is incomplete.

**Tech Stack:** MLIR C++ passes, Linalg/Tensor/Func dialect APIs, GoogleTest unit tests, lit/FileCheck tests, xvm CANN 9.1 verification.

---

## File Structure

- Modify: `include/Conversion/Ascend/Common/Attributes.h`
  - Add the shared attr name `kSymbolConstraintsAttr`.
- Create: `include/Conversion/Ascend/Common/SymbolConstraints.h`
  - Public helper API for `DimRef`, function-local value ordinals, serialized member parsing, class lookup, and structural verification.
- Create: `lib/Conversion/Ascend/Common/SymbolConstraints.cpp`
  - Helper implementation. Downstream layers must use this instead of parsing raw dictionary attrs.
- Create: `lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.h`
  - Normalize-private analysis entry points and proof-edge data structures.
- Create: `lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.cpp`
  - Weighted Union-Find and R1-R6 merge rules.
- Create: `lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.h`
  - Normalize-private verifier API.
- Create: `lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.cpp`
  - Attr structural checks plus recomputed R1-R6 coverage checks.
- Modify: `lib/Conversion/Ascend/Normalize/NormalizePass.cpp`
  - Invoke analysis, attach attr, run verifier, then stamp `ascend.normalized`.
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
  - Add new Common and Normalize sources.
- Modify: `test/unittests/Conversion/CMakeLists.txt`
  - Add `AscendSymbolConstraintsTest`.
- Create: `test/unittests/Conversion/AscendSymbolConstraintsTest.cpp`
  - Unit tests for value ordinal round-trip and attr validation.
- Create: `test/Conversion/ascend-normalize-symbol-constraints.mlir`
  - lit tests for R1-R6 attr emission and verifier failures.
- Modify: `test/Conversion/ascend-normalize.mlir`
  - Update existing CHECK lines to tolerate or check the new `ascend.symbol_constraints` attr on valid functions.

---

### Task 1: RED Unit Tests For Symbol Constraint Helpers

**Files:**
- Create: `test/unittests/Conversion/AscendSymbolConstraintsTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: Add the failing test target to CMake**

Add this block after `AscendCommonAttributesTest` in `test/unittests/Conversion/CMakeLists.txt`:

```cmake
add_executable(AscendSymbolConstraintsTest
  AscendSymbolConstraintsTest.cpp
)

target_include_directories(AscendSymbolConstraintsTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendSymbolConstraintsTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
  MLIRParser
)

add_dependencies(RuntimeUnitTests AscendSymbolConstraintsTest)

add_test(NAME AscendSymbolConstraintsTest
  COMMAND $<TARGET_FILE:AscendSymbolConstraintsTest>
)
```

- [ ] **Step 2: Write failing gtest coverage**

Create `test/unittests/Conversion/AscendSymbolConstraintsTest.cpp`:

```c++
//===- AscendSymbolConstraintsTest.cpp - Symbol constraint tests -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/SymbolConstraints.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

using namespace mlir;

std::unique_ptr<MLIRContext> createContext() {
  auto context = std::make_unique<MLIRContext>();
  context->loadDialect<arith::ArithDialect, func::FuncDialect,
                       linalg::LinalgDialect, tensor::TensorDialect>();
  return context;
}

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, StringRef text) {
  return parseSourceString<ModuleOp>(text, &context);
}

TEST(AscendSymbolConstraintsTest, BuildsFunctionLocalValueOrdinals) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @ordinals(%arg0: tensor<?x?xf16>,
                      %arg1: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %empty = tensor.empty(%d0, %d1) : tensor<?x?xf16>
    %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %out: f16):
      %sum = arith.addf %x, %y : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("ordinals");
  ASSERT_TRUE(func);

  mlir::ascend::symbol::ValueOrdinalMap ordinals =
      mlir::ascend::symbol::buildValueOrdinalMap(func);

  EXPECT_EQ(ordinals.lookup(func.getArgument(0)), 0);
  EXPECT_EQ(ordinals.lookup(func.getArgument(1)), 1);
  Operation *generic = nullptr;
  func.walk([&](linalg::GenericOp op) { generic = op; });
  ASSERT_NE(generic, nullptr);
  EXPECT_GT(ordinals.lookup(generic->getResult(0)), 1);

  FailureOr<Value> resolved =
      mlir::ascend::symbol::resolveValueOrdinal(func, ordinals,
                                                ordinals.lookup(generic->getResult(0)));
  ASSERT_FALSE(failed(resolved));
  EXPECT_EQ(*resolved, generic->getResult(0));
}

TEST(AscendSymbolConstraintsTest, RejectsDuplicateDimRefMembers) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @bad(%arg0: tensor<?xf16>)
      attributes {
        ascend.symbol_constraints = [
          {sym_name = "arg0_dim0", members = [
            {value = 0 : i64, dim = 0 : i64},
            {value = 0 : i64, dim = 0 : i64}
          ]}
        ]
      } {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("bad");
  ASSERT_TRUE(func);

  EXPECT_TRUE(failed(mlir::ascend::symbol::verifySymbolConstraintAttr(func)));
}

TEST(AscendSymbolConstraintsTest, LooksUpClassForResolvedDimRef) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @lookup(%arg0: tensor<?xf16>, %arg1: tensor<?xf16>)
      attributes {
        ascend.symbol_constraints = [
          {sym_name = "arg0_dim0", members = [
            {value = 0 : i64, dim = 0 : i64},
            {value = 1 : i64, dim = 0 : i64}
          ]}
        ]
      } {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("lookup");
  ASSERT_TRUE(func);

  FailureOr<mlir::ascend::symbol::SymbolConstraintTable> table =
      mlir::ascend::symbol::parseSymbolConstraintAttr(func);
  ASSERT_FALSE(failed(table));

  auto found = table->lookup({func.getArgument(1), 0});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->symName.getValue(), "arg0_dim0");
}

} // namespace
```

- [ ] **Step 3: Run the test and verify RED**

Run on xvm after local sync:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendSymbolConstraintsTest --gtest_filter=AscendSymbolConstraintsTest.*'
```

Expected: build fails because `Conversion/Ascend/Common/SymbolConstraints.h` does not exist.

- [ ] **Step 4: Commit RED tests**

```bash
git add test/unittests/Conversion/CMakeLists.txt test/unittests/Conversion/AscendSymbolConstraintsTest.cpp
git commit -m "test: add symbol constraint helper red tests"
```

---

### Task 2: GREEN Shared Symbol Constraint Helpers

**Files:**
- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Create: `include/Conversion/Ascend/Common/SymbolConstraints.h`
- Create: `lib/Conversion/Ascend/Common/SymbolConstraints.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Add the shared attr name**

In `include/Conversion/Ascend/Common/Attributes.h`, add this constant next to
`kNormalizedAttr`:

```c++
inline constexpr llvm::StringLiteral kSymbolConstraintsAttr =
    "ascend.symbol_constraints";
```

- [ ] **Step 2: Add the public helper header**

Create `include/Conversion/Ascend/Common/SymbolConstraints.h`:

```c++
//===- SymbolConstraints.h - Ascend symbol constraints -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::ascend::symbol {

struct DimRef {
  Value value;
  int64_t dim = -1;

  friend bool operator==(const DimRef &lhs, const DimRef &rhs) {
    return lhs.value == rhs.value && lhs.dim == rhs.dim;
  }
};

struct SerializedDimRef {
  int64_t valueOrdinal = -1;
  int64_t dim = -1;
};

struct SymbolConstraintClass {
  StringAttr symName;
  SmallVector<DimRef, 4> members;
};

using ValueOrdinalMap = llvm::DenseMap<Value, int64_t>;

struct SymbolConstraintTable {
  SmallVector<SymbolConstraintClass, 4> classes;
  const SymbolConstraintClass *lookup(DimRef ref) const;
  bool areEquivalent(DimRef lhs, DimRef rhs) const;
};

ValueOrdinalMap buildValueOrdinalMap(func::FuncOp func);
FailureOr<Value> resolveValueOrdinal(func::FuncOp func,
                                     const ValueOrdinalMap &ordinals,
                                     int64_t ordinal);
FailureOr<SerializedDimRef> serializeDimRef(func::FuncOp func,
                                            const ValueOrdinalMap &ordinals,
                                            DimRef ref);
DictionaryAttr buildSerializedDimRefAttr(MLIRContext *context,
                                         SerializedDimRef ref);
DictionaryAttr buildSymbolClassAttr(MLIRContext *context, StringRef symName,
                                    ArrayRef<SerializedDimRef> members);
ArrayAttr buildSymbolConstraintAttr(MLIRContext *context,
                                    ArrayRef<DictionaryAttr> classes);
FailureOr<SymbolConstraintTable> parseSymbolConstraintAttr(func::FuncOp func);
LogicalResult verifySymbolConstraintAttr(func::FuncOp func);

} // namespace mlir::ascend::symbol

namespace llvm {
template <>
struct DenseMapInfo<mlir::ascend::symbol::DimRef> {
  static mlir::ascend::symbol::DimRef getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey(), -1};
  }
  static mlir::ascend::symbol::DimRef getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey(), -1};
  }
  static unsigned getHashValue(const mlir::ascend::symbol::DimRef &ref) {
    return hash_combine(ref.value, ref.dim);
  }
  static bool isEqual(const mlir::ascend::symbol::DimRef &lhs,
                      const mlir::ascend::symbol::DimRef &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H
```

- [ ] **Step 3: Implement helper behavior**

Create `lib/Conversion/Ascend/Common/SymbolConstraints.cpp` with these core
functions. Keep parsing strict and emit errors on the owning function.

```c++
//===- SymbolConstraints.cpp - Ascend symbol constraints ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/SymbolConstraints.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::ascend::symbol {
namespace {

bool isRankedTensor(Value value) {
  return isa<RankedTensorType>(value.getType());
}

RankedTensorType getRankedTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

IntegerAttr getI64Attr(DictionaryAttr dict, StringRef name) {
  Attribute attr = dict.get(name);
  return dyn_cast_or_null<IntegerAttr>(attr);
}

StringAttr getStringAttr(DictionaryAttr dict, StringRef name) {
  Attribute attr = dict.get(name);
  return dyn_cast_or_null<StringAttr>(attr);
}

ArrayAttr getArrayAttr(DictionaryAttr dict, StringRef name) {
  Attribute attr = dict.get(name);
  return dyn_cast_or_null<ArrayAttr>(attr);
}

} // namespace

const SymbolConstraintClass *SymbolConstraintTable::lookup(DimRef ref) const {
  for (const SymbolConstraintClass &klass : classes)
    for (DimRef member : klass.members)
      if (member == ref)
        return &klass;
  return nullptr;
}

bool SymbolConstraintTable::areEquivalent(DimRef lhs, DimRef rhs) const {
  const SymbolConstraintClass *lhsClass = lookup(lhs);
  return lhsClass && lhsClass == lookup(rhs);
}

ValueOrdinalMap buildValueOrdinalMap(func::FuncOp func) {
  ValueOrdinalMap ordinals;
  int64_t nextOrdinal = 0;
  for (BlockArgument arg : func.getArguments())
    if (isRankedTensor(arg))
      ordinals.try_emplace(arg, nextOrdinal++);

  func.walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (isRankedTensor(result))
        ordinals.try_emplace(result, nextOrdinal++);
  });
  return ordinals;
}

FailureOr<Value> resolveValueOrdinal(func::FuncOp func,
                                     const ValueOrdinalMap &ordinals,
                                     int64_t ordinal) {
  for (const auto &entry : ordinals)
    if (entry.second == ordinal)
      return entry.first;
  return failure();
}

FailureOr<SerializedDimRef> serializeDimRef(func::FuncOp func,
                                            const ValueOrdinalMap &ordinals,
                                            DimRef ref) {
  auto it = ordinals.find(ref.value);
  if (it == ordinals.end())
    return failure();
  RankedTensorType type = getRankedTensorType(ref.value);
  if (!type || ref.dim < 0 || ref.dim >= type.getRank())
    return failure();
  return SerializedDimRef{it->second, ref.dim};
}

DictionaryAttr buildSerializedDimRefAttr(MLIRContext *context,
                                         SerializedDimRef ref) {
  Builder builder(context);
  return builder.getDictionaryAttr({
      builder.getNamedAttr("value", builder.getI64IntegerAttr(ref.valueOrdinal)),
      builder.getNamedAttr("dim", builder.getI64IntegerAttr(ref.dim)),
  });
}

DictionaryAttr buildSymbolClassAttr(MLIRContext *context, StringRef symName,
                                    ArrayRef<SerializedDimRef> members) {
  Builder builder(context);
  SmallVector<Attribute> memberAttrs;
  for (SerializedDimRef member : members)
    memberAttrs.push_back(buildSerializedDimRefAttr(context, member));
  return builder.getDictionaryAttr({
      builder.getNamedAttr("sym_name", builder.getStringAttr(symName)),
      builder.getNamedAttr("members", builder.getArrayAttr(memberAttrs)),
  });
}

ArrayAttr buildSymbolConstraintAttr(MLIRContext *context,
                                    ArrayRef<DictionaryAttr> classes) {
  SmallVector<Attribute> classAttrs;
  for (DictionaryAttr klass : classes)
    classAttrs.push_back(klass);
  return ArrayAttr::get(context, classAttrs);
}

FailureOr<SymbolConstraintTable> parseSymbolConstraintAttr(func::FuncOp func) {
  auto classesAttr =
      dyn_cast_or_null<ArrayAttr>(func->getAttr(kSymbolConstraintsAttr));
  if (!classesAttr) {
    func.emitError() << "missing " << kSymbolConstraintsAttr;
    return failure();
  }

  ValueOrdinalMap ordinals = buildValueOrdinalMap(func);
  llvm::DenseSet<StringAttr> seenNames;
  llvm::DenseSet<DimRef> seenMembers;
  SymbolConstraintTable table;

  for (Attribute classRaw : classesAttr) {
    auto classDict = dyn_cast<DictionaryAttr>(classRaw);
    if (!classDict)
      return func.emitError("symbol constraint class must be a dictionary"),
             failure();
    StringAttr symName = getStringAttr(classDict, "sym_name");
    ArrayAttr members = getArrayAttr(classDict, "members");
    if (!symName || symName.getValue().empty())
      return func.emitError("symbol constraint class needs non-empty sym_name"),
             failure();
    if (!seenNames.insert(symName).second)
      return func.emitError() << "duplicate symbol constraint name "
                              << symName.getValue(), failure();
    if (!members || members.empty())
      return func.emitError("symbol constraint class needs non-empty members"),
             failure();

    SymbolConstraintClass parsed;
    parsed.symName = symName;
    for (Attribute memberRaw : members) {
      auto memberDict = dyn_cast<DictionaryAttr>(memberRaw);
      if (!memberDict)
        return func.emitError("symbol constraint member must be a dictionary"),
               failure();
      IntegerAttr valueAttr = getI64Attr(memberDict, "value");
      IntegerAttr dimAttr = getI64Attr(memberDict, "dim");
      if (!valueAttr || !dimAttr)
        return func.emitError("symbol constraint member needs value and dim"),
               failure();
      FailureOr<Value> value =
          resolveValueOrdinal(func, ordinals, valueAttr.getInt());
      if (failed(value))
        return func.emitError("symbol constraint member has invalid value ordinal"),
               failure();
      RankedTensorType type = getRankedTensorType(*value);
      int64_t dim = dimAttr.getInt();
      if (!type || dim < 0 || dim >= type.getRank())
        return func.emitError("symbol constraint member has invalid dim"),
               failure();
      DimRef ref{*value, dim};
      if (!seenMembers.insert(ref).second)
        return func.emitError("duplicate DimRef in symbol constraints"),
               failure();
      parsed.members.push_back(ref);
    }
    table.classes.push_back(std::move(parsed));
  }
  return table;
}

LogicalResult verifySymbolConstraintAttr(func::FuncOp func) {
  return succeeded(parseSymbolConstraintAttr(func)) ? success() : failure();
}

} // namespace mlir::ascend::symbol
```

- [ ] **Step 4: Add the Common source to the library**

Modify `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
set(ASCEND_COMMON_SOURCES
  Common/SymbolConstraints.cpp
)

add_mlir_library(AscendConversion
  ${ASCEND_COMMON_SOURCES}
  ${ASCEND_NORMALIZE_SOURCES}
  ${ASCEND_KERNELIZE_SOURCES}
  ${ASCEND_SCHEDULE_SOURCES}
  ${ASCEND_REALIZE_SOURCES}
  ${ASCEND_TRANSLATE_KERNELIR_SOURCES}
  ${ASCEND_TRANSLATE_PREEMIT_SOURCES}
  ${ASCEND_DEBUG_SOURCES}
```

- [ ] **Step 5: Run helper tests and verify GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendSymbolConstraintsTest --gtest_filter=AscendSymbolConstraintsTest.*'
```

Expected: all `AscendSymbolConstraintsTest.*` tests pass.

- [ ] **Step 6: Commit helper implementation**

```bash
git add include/Conversion/Ascend/Common/Attributes.h include/Conversion/Ascend/Common/SymbolConstraints.h lib/Conversion/Ascend/Common/SymbolConstraints.cpp lib/Conversion/Ascend/CMakeLists.txt
git commit -m "feat: add symbol constraint attr helpers"
```

---

### Task 3: RED lit Tests For Normalize R1-R6

**Files:**
- Create: `test/Conversion/ascend-normalize-symbol-constraints.mlir`

- [ ] **Step 1: Write lit tests for attr emission**

Create `test/Conversion/ascend-normalize-symbol-constraints.mlir`:

```mlir
// RUN: sed -n '/\/\/ R1-BEGIN/,/\/\/ R1-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R1
// RUN: sed -n '/\/\/ R2-BEGIN/,/\/\/ R2-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R2
// RUN: sed -n '/\/\/ R3-BEGIN/,/\/\/ R3-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R3
// RUN: sed -n '/\/\/ R4-BEGIN/,/\/\/ R4-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R4
// RUN: sed -n '/\/\/ R4-NEG-BEGIN/,/\/\/ R4-NEG-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R4-NEG
// RUN: sed -n '/\/\/ R5-BEGIN/,/\/\/ R5-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R5
// RUN: sed -n '/\/\/ R6-BEGIN/,/\/\/ R6-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R6
// RUN: sed -n '/\/\/ STATIC-BEGIN/,/\/\/ STATIC-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=STATIC

// R1-BEGIN
func.func @r1_generic_matmul_like(%lhs: tensor<?x?xf16>,
                                  %rhs: tensor<?x?xf16>,
                                  %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %acc: f16):
      %product = arith.mulf %x, %y : f16
      %sum = arith.addf %acc, %product : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
// R1-END

// R1-LABEL: func.func @r1_generic_matmul_like
// R1: ascend.symbol_constraints
// R1-DAG: sym_name = "arg0_dim0"
// R1-DAG: sym_name = "arg0_dim1"
// R1-DAG: sym_name = "arg1_dim1"

// R2-BEGIN
func.func @r2_named_matmul(%lhs: tensor<?x?xf16>,
                           %rhs: tensor<?x?xf16>,
                           %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
                     outs(%out : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
// R2-END

// R2-LABEL: func.func @r2_named_matmul
// R2: ascend.symbol_constraints
// R2-DAG: sym_name = "arg0_dim0"
// R2-DAG: sym_name = "arg0_dim1"
// R2-DAG: sym_name = "arg1_dim1"

// R3-BEGIN
func.func @r3_producer_consumer(%arg0: tensor<?x?xf16>,
                                %out0: tensor<?x?xf16>,
                                %out1: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%out0 : tensor<?x?xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<?x?xf16>)
      outs(%out1 : tensor<?x?xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  return %1 : tensor<?x?xf16>
}
// R3-END

// R3-LABEL: func.func @r3_producer_consumer
// R3: ascend.symbol_constraints
// R3-DAG: sym_name = "arg0_dim0"
// R3-DAG: sym_name = "arg0_dim1"

// R4-BEGIN
func.func @r4_extract_full_slice(%src: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %src, %c0 : tensor<?x?xf16>
  %c1 = arith.constant 1 : index
  %n = tensor.dim %src, %c1 : tensor<?x?xf16>
  %slice = tensor.extract_slice %src[%c0, %c0][%m, %n][1, 1]
      : tensor<?x?xf16> to tensor<?x?xf16>
  return %slice : tensor<?x?xf16>
}
// R4-END

// R4-LABEL: func.func @r4_extract_full_slice
// R4: ascend.symbol_constraints
// R4-DAG: sym_name = "arg0_dim0"
// R4-DAG: sym_name = "arg0_dim1"

// R4-NEG-BEGIN
func.func @r4_strided_slice_does_not_merge(%src: tensor<?xf16>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %src, %c0 : tensor<?xf16>
  %slice = tensor.extract_slice %src[%c0][%m][2]
      : tensor<?xf16> to tensor<?xf16>
  return %slice : tensor<?xf16>
}
// R4-NEG-END

// R4-NEG-LABEL: func.func @r4_strided_slice_does_not_merge
// R4-NEG: ascend.symbol_constraints = []
// R4-NEG-NOT: sym_name = "arg0_dim0"

// R5-BEGIN
func.func @r5_tensor_dim_empty(%src: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %src, %c0 : tensor<?x?xf16>
  %n = tensor.dim %src, %c1 : tensor<?x?xf16>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf16>
  return %empty : tensor<?x?xf16>
}
// R5-END

// R5-LABEL: func.func @r5_tensor_dim_empty
// R5: ascend.symbol_constraints
// R5-DAG: sym_name = "arg0_dim0"
// R5-DAG: sym_name = "arg0_dim1"

// R6-BEGIN
func.func @r6_linalg_broadcast(%input: tensor<?x?xf16>,
                               %out: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = linalg.broadcast ins(%input : tensor<?x?xf16>)
      outs(%out : tensor<?x?x?xf16>) dimensions = [0]
  return %0 : tensor<?x?x?xf16>
}
// R6-END

// R6-LABEL: func.func @r6_linalg_broadcast
// R6: ascend.symbol_constraints
// R6-DAG: sym_name = "arg0_dim0"
// R6-DAG: sym_name = "arg0_dim1"

// STATIC-BEGIN
func.func @static_dims_are_not_members(%arg0: tensor<4x?xf16>,
                                       %out: tensor<4x?xf16>) -> tensor<4x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<4x?xf16>)
      outs(%out : tensor<4x?xf16>) {
    ^bb0(%x: f16, %outv: f16):
      linalg.yield %x : f16
    } -> tensor<4x?xf16>
  return %0 : tensor<4x?xf16>
}
// STATIC-END

// STATIC-LABEL: func.func @static_dims_are_not_members
// STATIC: ascend.symbol_constraints
// STATIC-NOT: sym_name = "arg0_dim0"
// STATIC: sym_name = "arg0_dim1"
```

- [ ] **Step 2: Add malformed attr verifier tests**

Append this negative test to the same file:

```mlir
// RUN: sed -n '/\/\/ BAD-BEGIN/,/\/\/ BAD-END/p' %s | not afir-opt --ascend-normalize 2>&1 | FileCheck %s --check-prefix=BAD

// BAD-BEGIN
func.func @bad_existing_symbol_attr(%arg0: tensor<?xf16>)
    attributes {
      ascend.symbol_constraints = [
        {sym_name = "arg0_dim0", members = [
          {value = 0 : i64, dim = 0 : i64},
          {value = 0 : i64, dim = 0 : i64}
        ]}
      ]
    } {
  return
}
// BAD-END

// BAD: duplicate DimRef in symbol constraints
```

- [ ] **Step 3: Run lit test and verify RED**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize-symbol-constraints.mlir'
```

Expected: positive checks fail because Normalize does not emit `ascend.symbol_constraints`.

- [ ] **Step 4: Commit RED lit tests**

```bash
git add test/Conversion/ascend-normalize-symbol-constraints.mlir
git commit -m "test: add normalize symbol constraint red tests"
```

---

### Task 4: GREEN Normalize Symbol Equivalence Analysis

**Files:**
- Create: `lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.h`
- Create: `lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Normalize/NormalizePass.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Add the analysis header**

Create `lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.h`:

```c++
//===- SymbolEquivalenceAnalysis.h - Normalize symbol analysis -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend::normalize {

struct SymbolEqualityProof {
  symbol::DimRef lhs;
  symbol::DimRef rhs;
  StringRef rule;
};

struct SymbolEquivalenceResult {
  ArrayAttr attr;
  SmallVector<SymbolEqualityProof, 16> proofs;
};

FailureOr<SymbolEquivalenceResult> analyzeSymbolEquivalence(func::FuncOp func);

} // namespace mlir::ascend::normalize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H
```

- [ ] **Step 2: Implement the weighted Union-Find skeleton**

In `SymbolEquivalenceAnalysis.cpp`, add an internal class with parent, rank,
and zero weights. R1-R6 use equality edges, so every merge uses weight zero.

```c++
class DimUnionFind {
public:
  void add(symbol::DimRef ref);
  void uniteEqual(symbol::DimRef lhs, symbol::DimRef rhs, StringRef rule);
  SmallVector<SmallVector<symbol::DimRef, 4>, 8> buildClasses();
  ArrayRef<SymbolEqualityProof> getProofs() const { return proofs; }

private:
  struct Node {
    symbol::DimRef parent;
    int64_t rank = 0;
    int64_t weightToParent = 0;
  };

  symbol::DimRef find(symbol::DimRef ref);

  llvm::DenseMap<symbol::DimRef, Node> nodes;
  SmallVector<SymbolEqualityProof, 16> proofs;
};
```

Implementation rule: `uniteEqual` records a proof edge even if both refs already
belong to the same component. The verifier uses these proof edges for coverage.

- [ ] **Step 3: Add DimRef creation helpers**

In the same file, add helpers:

```c++
static RankedTensorType getRankedTensor(Value value);
static void addAllRankedTensorDims(func::FuncOp func, DimUnionFind &uf);
static FailureOr<symbol::DimRef> makeDimRef(Value value, int64_t dim);
static bool isDynamicDim(symbol::DimRef ref);
static bool hasStaticDimValue(symbol::DimRef ref, int64_t value);
```

Rules:

- Only ranked tensor values become nodes.
- Static dimensions are added as nodes only when they participate in a proof
  edge; they are filtered out before attr serialization.
- Invalid dim indexes return `failure()`.

- [ ] **Step 4: Implement R1 generic indexing maps**

Use `linalg::GenericOp` and its indexing maps. For each iterator position, find
projected tensor dimensions for each input and init/result map. Merge pairs that
project the same iterator to concrete result dimensions.

Implementation shape:

```c++
static void applyGenericIndexingRule(linalg::GenericOp op, DimUnionFind &uf) {
  SmallVector<Value> values;
  SmallVector<AffineMap> maps = llvm::to_vector(op.getIndexingMapsArray());
  for (OpOperand *input : op.getDpsInputOperands())
    values.push_back(input->get());
  for (OpOperand *init : op.getDpsInitOperands())
    values.push_back(init->get());

  for (unsigned iter = 0; iter < op.getNumLoops(); ++iter) {
    SmallVector<symbol::DimRef, 4> refsForIterator;
    for (auto [value, map] : llvm::zip_equal(values, maps)) {
      for (auto [resultIndex, expr] : llvm::enumerate(map.getResults())) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr || dimExpr.getPosition() != iter)
          continue;
        FailureOr<symbol::DimRef> ref = makeDimRef(value, resultIndex);
        if (succeeded(ref))
          refsForIterator.push_back(*ref);
      }
    }
    for (unsigned i = 1; i < refsForIterator.size(); ++i)
      uf.uniteEqual(refsForIterator[0], refsForIterator[i], "R1");
  }
}
```

- [ ] **Step 5: Implement R2 named contraction rules**

Cover `linalg::MatmulOp` and `linalg::BatchMatmulOp` with static semantic
rules. Use op operands and results, not generalized named-op lowering.

Matmul edges:

```text
lhs.0 == result.0
lhs.1 == rhs.0
rhs.1 == result.1
```

Batch matmul edges:

```text
lhs.0 == rhs.0 == result.0
lhs.1 == result.1
lhs.2 == rhs.1
rhs.2 == result.2
```

- [ ] **Step 6: Implement R3 producer-consumer identity**

Do not create fake operand-context `DimRef`s. In MLIR, an operand is the same
SSA `Value` as its producer result, so `DimRef(Value, dim)` already captures
R3. Add proof edges for each ranked tensor operand whose defining op exists:

```c++
if (Operation *def = operand.getDefiningOp())
  for each dim in ranked operand type:
    uf.uniteEqual({operand, dim}, {def->getResult(resultNumber), dim}, "R3");
```

This records R3 coverage while preserving the V2 `Value + dim` semantic model.

- [ ] **Step 7: Implement R4 extract_slice full-size rules**

For `tensor::ExtractSliceOp`, iterate source/result rank positions. Merge when:

- static stride is one,
- result dim corresponds to a non-dropped source dim,
- size is a `tensor.dim(source, sourceDim)` result, or
- static size equals the static source dimension and is not one.

Use MLIR mixed offsets/sizes/strides APIs and reject ambiguous affine/index
arithmetic.

- [ ] **Step 8: Implement R5 tensor.dim def-use rules**

Collect a map:

```c++
llvm::DenseMap<Value, symbol::DimRef> tensorDimResultToSourceDim;
```

Populate it from `tensor::DimOp` only when the index is a constant. Consume it
for:

- `tensor::EmptyOp`: dynamic size operand N maps to the Nth dynamic result dim.
- `tensor::ExtractSliceOp`: dynamic size operands that prove source dimension
  equality.

Merge only direct shape-carrying uses. Do not infer equality through arithmetic.

- [ ] **Step 9: Implement R6 linalg.broadcast**

For `linalg::BroadcastOp`, read the input, init/result, and `dimensions`
attribute. The listed output dimensions are broadcast dimensions. For each input
dimension, merge it with the next output dimension not listed in
`dimensions`.

- [ ] **Step 10: Serialize classes**

After all merges:

- Drop static constant dimensions from serialized members.
- Drop singleton classes that have no proof edge and no downstream value.
- Name classes by first function argument member, then `sym_<ordinal>`.
- Sort members by function-local value ordinal then dim.
- Sort classes by first member ordinal then first member dim.

Use `symbol::buildSymbolClassAttr` and `symbol::buildSymbolConstraintAttr`.

- [ ] **Step 11: Wire analysis into Normalize pass**

In `NormalizePass.cpp`, after the dialect whitelist succeeds and before setting
`ascend.normalized`, add:

```c++
for (func::FuncOp func : module.getOps<func::FuncOp>()) {
  if (func->hasAttr(::mlir::ascend::kSymbolConstraintsAttr) &&
      failed(symbol::verifySymbolConstraintAttr(func))) {
    signalPassFailure();
    return;
  }

  FailureOr<normalize::SymbolEquivalenceResult> result =
      normalize::analyzeSymbolEquivalence(func);
  if (failed(result)) {
    signalPassFailure();
    return;
  }
  func->setAttr(::mlir::ascend::kSymbolConstraintsAttr, result->attr);
}
```

- [ ] **Step 12: Add sources to CMake**

Modify `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
set(ASCEND_NORMALIZE_SOURCES
  Normalize/NormalizePass.cpp
  Normalize/SymbolEquivalenceAnalysis.cpp
)
```

- [ ] **Step 13: Run lit positives and verify GREEN for attr emission**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize-symbol-constraints.mlir'
```

Expected at this task boundary: positive R1-R6 checks pass; the malformed attr
negative may still fail for the wrong reason until Task 5.

- [ ] **Step 14: Commit analysis implementation**

```bash
git add lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.h lib/Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.cpp lib/Conversion/Ascend/Normalize/NormalizePass.cpp lib/Conversion/Ascend/CMakeLists.txt
git commit -m "feat: emit normalize symbol constraints"
```

---

### Task 5: GREEN Entry Normalization Verifier

**Files:**
- Create: `lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.h`
- Create: `lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.cpp`
- Modify: `lib/Conversion/Ascend/Normalize/NormalizePass.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Add verifier header**

Create `lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.h`:

```c++
//===- EntryNormalizationVerifier.h - Normalize verifier -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_ENTRYNORMALIZATIONVERIFIER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_ENTRYNORMALIZATIONVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::normalize {

LogicalResult verifyEntryNormalization(ModuleOp module);

} // namespace mlir::ascend::normalize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_ENTRYNORMALIZATIONVERIFIER_H
```

- [ ] **Step 2: Implement structural and coverage checks**

Create `EntryNormalizationVerifier.cpp`:

```c++
//===- EntryNormalizationVerifier.cpp - Normalize verifier ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Normalize/EntryNormalizationVerifier.h"

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "Conversion/Ascend/Normalize/SymbolEquivalenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace mlir::ascend::normalize {

LogicalResult verifyEntryNormalization(ModuleOp module) {
  LogicalResult result = success();
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    FailureOr<symbol::SymbolConstraintTable> table =
        symbol::parseSymbolConstraintAttr(func);
    if (failed(table)) {
      result = failure();
      continue;
    }

    FailureOr<SymbolEquivalenceResult> expected =
        analyzeSymbolEquivalence(func);
    if (failed(expected)) {
      func.emitError("failed to recompute symbol equivalence proofs");
      result = failure();
      continue;
    }

    for (const SymbolEqualityProof &proof : expected->proofs) {
      if (!table->areEquivalent(proof.lhs, proof.rhs)) {
        func.emitError() << "symbol constraints missing " << proof.rule
                         << " equality proof";
        result = failure();
      }
    }
  }
  return result;
}

} // namespace mlir::ascend::normalize
```

- [ ] **Step 3: Wire verifier into Normalize pass**

In `NormalizePass.cpp`, after all functions receive `ascend.symbol_constraints`
and before `ascend.normalized` is set:

```c++
if (failed(normalize::verifyEntryNormalization(module))) {
  signalPassFailure();
  return;
}
```

The verifier must call `analyzeSymbolEquivalence(func)` only as a pure
recomputation step. It must not set, remove, or rewrite attributes.

- [ ] **Step 4: Add verifier source to CMake**

Modify `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
set(ASCEND_NORMALIZE_SOURCES
  Normalize/EntryNormalizationVerifier.cpp
  Normalize/NormalizePass.cpp
  Normalize/SymbolEquivalenceAnalysis.cpp
)
```

- [ ] **Step 5: Run negative lit and verify GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize-symbol-constraints.mlir test/Conversion/ascend-normalize.mlir'
```

Expected: symbol constraint lit tests and existing normalize tests pass.

- [ ] **Step 6: Commit verifier**

```bash
git add lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.h lib/Conversion/Ascend/Normalize/EntryNormalizationVerifier.cpp lib/Conversion/Ascend/Normalize/NormalizePass.cpp lib/Conversion/Ascend/CMakeLists.txt test/Conversion/ascend-normalize.mlir
git commit -m "feat: verify normalize symbol constraints"
```

---

### Task 6: Regression Compatibility With Existing Pipeline Tests

**Files:**
- Modify only tests that fail because CHECK lines need to accept the new
  `ascend.symbol_constraints` function attr.

- [ ] **Step 1: Run focused conversion tests likely to print function attrs**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize.mlir test/Conversion/ascend-kernelize-mvp.mlir test/Conversion/ascend-schedule-mvp.mlir test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir'
```

Expected: failures, if any, are FileCheck attr-format mismatches caused by the
new function-level `ascend.symbol_constraints` attr.

- [ ] **Step 2: Update only attr-sensitive CHECK lines**

For each affected test, prefer matching required stable attrs instead of exact
full attribute dictionaries. Example pattern:

```mlir
// CHECK-LABEL: func.func @valid
// CHECK-SAME: ascend.normalized = true
// CHECK-SAME: ascend.symbol_constraints
```

Do not relax checks for kernel roles, schedule metadata, memory placement, or
lowering output.

- [ ] **Step 3: Re-run focused conversion tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize.mlir test/Conversion/ascend-normalize-symbol-constraints.mlir test/Conversion/ascend-kernelize-mvp.mlir test/Conversion/ascend-schedule-mvp.mlir test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir'
```

Expected: all focused conversion tests pass.

- [ ] **Step 4: Commit test compatibility updates**

```bash
git add test/Conversion
git commit -m "test: accept normalize symbol constraint attrs"
```

---

### Task 7: Final xvm Verification

**Files:**
- No source edits unless verification exposes a bug in this feature.

- [ ] **Step 1: Build on xvm**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build'
```

Expected: build succeeds.

- [ ] **Step 2: Run unit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendSymbolConstraintsTest --gtest_filter=AscendSymbolConstraintsTest.* && ./build/bin/AscendCommonAttributesTest --gtest_filter=AscendCommonAttributesTest.*'
```

Expected: both test binaries pass.

- [ ] **Step 3: Run focused lit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-normalize.mlir test/Conversion/ascend-normalize-symbol-constraints.mlir test/Conversion/ascend-kernelize-mvp.mlir test/Conversion/ascend-schedule-mvp.mlir'
```

Expected: all focused lit tests pass.

- [ ] **Step 4: Run normal conversion regression smoke**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion'
```

Expected: conversion lit suite passes or only pre-existing unsupported tests are
reported as unsupported.

- [ ] **Step 5: Run git hygiene locally**

```bash
git status --short
git diff --check
```

Expected: no unstaged changes except intentional feature files before final
commit; `git diff --check` prints no whitespace errors.

- [ ] **Step 6: Final commit if verification required fixes**

If Task 7 changed files, commit only this feature's files:

```bash
git add include/Conversion/Ascend/Common/Attributes.h include/Conversion/Ascend/Common/SymbolConstraints.h lib/Conversion/Ascend/Common/SymbolConstraints.cpp lib/Conversion/Ascend/CMakeLists.txt lib/Conversion/Ascend/Normalize test/Conversion test/unittests/Conversion
git commit -m "fix: complete symbol equivalence verification"
```

Expected: final branch history contains the RED tests, helper implementation,
Normalize analysis, verifier, and focused compatibility fixes as separate
commits.
