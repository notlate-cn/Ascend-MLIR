include "mlir/Pass/PassBase.td"

def ConvertTmTensorAttention : Pass<"convert-tm-tensor-attention", "mlir::ModuleOp"> {
  let summary = "Lower tm_tensor.attention to func.call @__aclnn_flash_attention";
  let constructor = "mlir::afir::createConvertTmTensorAttentionPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def RemoveCfAssert : Pass<"remove-cf-assert", "mlir::func::FuncOp"> {
  let summary = "Remove cf.assert ops introduced by the torch frontend";
  let constructor = "mlir::afir::createRemoveCfAssertPass()";
}