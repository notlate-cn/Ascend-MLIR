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

def AFIRAddAxisPass : Pass<"afir-add-axis", "mlir::ModuleOp"> {
  let summary = "Add Axis for AFIR operations";
  let description = [{
    This pass add axis fpr AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRAddAxisPass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def LinalgAddBroadcastPass : Pass<"linalg-add-broadcast", "mlir::ModuleOp"> {
  let summary = "Add broadcast for linalg Op";
  let description = [{
    This pass Add broadcast for linalg Op.
  }];
  let constructor = "mlir::createLinalgAddBroadcastPass()";
}

def LinalgInferShapePass : Pass<"linalg-infer-shape", "mlir::ModuleOp"> {
  let summary = "infer static shape for linalg operations";
  let description = [{
    This pass infer static shape for linalg operations.
  }];
  let constructor = "mlir::createLinalgInferShapePass()";
}

def LinalgMark : Pass<"linalg-mark", "mlir::ModuleOp"> {
  let summary = "mark linalg named op";
  let description = [{
    This pass add axis fpr AFIR operations.
  }];
  let constructor = "mlir::createLinalgMarkPass()";
}

#endif // AFIR_TRANSFORMS_PASSES
