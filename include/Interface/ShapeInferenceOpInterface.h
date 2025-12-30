/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ShapeInferenceOpInterface.h - Definition for ShapeInference ----===//
//
// Copyright 2024 The Ascend-MLIR Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceOpInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_SHAPE_INFERENCE_INTERFACE_H
#define ASCEND_MLIR_SHAPE_INFERENCE_INTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "Interface/ShapeInferenceOpInterface.h.inc"

#endif  // ASCEND_MLIR_SHAPE_INFERENCE_INTERFACE_H