//===- Passes.td - AFIR dialect passes ---------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TRANSFORMS_PASSES
#define AFIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AFIRCanonicalizePass : Pass<"afir-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize AFIR operations";
  let description = [{
    This pass performs canonicalization transformations on AFIR operations,
    including constant folding and algebraic simplifications.
  }];
  let constructor = "mlir::afir::createAFIRCanonicalizePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def AFIRShapeInferencePass : Pass<"afir-shape-inference", "mlir::ModuleOp"> {
  let summary = "Infer shapes for AFIR operations";
  let description = [{
    This pass infers and propagates shape information through AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRShapeInferencePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

#endif // AFIR_TRANSFORMS_PASSES
