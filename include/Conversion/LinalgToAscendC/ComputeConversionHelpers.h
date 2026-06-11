/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under terms and conditions of the CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to LICENSE in the root of the
 * software repository for the full text of the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#ifndef CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONHELPERS_H
#define CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONHELPERS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
namespace afir {

// Stateless helpers for the linalg→AscendC compute lowering.  These close over
// no convertCompute state (they were `[]` lambdas), so they live here as free
// functions to keep ComputeConversion.cpp focused on the lowering itself.

// Copy the "ascendc.unit" string attribute from `src` to `dst`, if present.
void copyAscendCUnitAttr(Operation *src, Operation *dst);

// If `value` is a min(step, tail) clamp against an enclosing loop's step,
// return the loop step operand (the full, unclamped tile size); else `value`.
Value getEnclosingLoopStepBound(Value value, Operation *anchor);

// Return the nearest enclosing scf::ForOp of `op`, or nullptr.
scf::ForOp getEnclosingFor(Operation *op);

// Classification of an input indexing map relative to the iteration space.
struct IndexingMapAnalysis {
  enum class Kind {
    Identity,           // (d0,d1)->(d0,d1): direct read
    PureBroadcast,      // (d0,d1)->(d0): some dims absent, no reordering
    PureTranspose,      // (d0,d1)->(d1,d0): all dims present, permuted
    BroadcastTranspose, // (d0,d1)->(d1,0): constants + reordering
  };
  Kind kind;
  SmallVector<int64_t> permutation;   // valid for PureTranspose, BroadcastTranspose
  SmallVector<int64_t> broadcastDims; // iteration dims absent from output
};

// Analyze an input indexing map to classify how the input is accessed relative
// to the iteration space of rank `iterRank`.
IndexingMapAnalysis analyzeIndexingMap(AffineMap map, unsigned iterRank);

// True when `map` projects away at least one of the `iterRank` iteration dims.
bool isBroadcastMap(AffineMap map, unsigned iterRank);

// C++ scalar type name for a verbatim template.
std::string cppScalarName(Type t);

// Detect a standalone transpose generic (single input with a non-identity
// permutation map, identity output map, yield-only body).
bool isTransposeGeneric(linalg::GenericOp op);

// True when a transpose with element type `elemType` and permutation `perm` is
// correctly realizable by AscendC::Transpose (a 16x16 `vtranspose`): 16-bit
// dtype and a rank-2 [1,0] swap.  See ComputeConversion.cpp for the rationale.
bool transposeSupportedByIntrinsic(Type elemType, ArrayRef<int64_t> perm);

// True when a transpose is realizable by the AF ConfusionTranspose path
// (codegen::AfirConfusionTranspose2D, built on AscendC::TransDataTo5HD): f16 OR
// f32, rank-2 [1,0] swap, ANY [H,W] size (the 16x16 fractal blocking is internal,
// unlike the bare vtranspose). Replaces transposeSupportedByIntrinsic at the
// on-chip transpose emit sites.
bool transposeSupportedByConfusion(Type elemType, ArrayRef<int64_t> perm);

// Detect index_select (column) gather — stamped {gather_dim} by marking.
bool isIndexSelectGeneric(linalg::GenericOp op);

// Detect embedding (row) gather — stamped {embedding_dim} by marking.
bool isEmbeddingGeneric(linalg::GenericOp op);

} // namespace afir
} // namespace mlir

#endif // CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONHELPERS_H
