#ifndef ASCEND_KERNELIZE_OP_INTERFACES
#define ASCEND_KERNELIZE_OP_INTERFACES

include "mlir/IR/OpBase.td"

def KernelizeSemanticOpInterface
    : OpInterface<"KernelizeSemanticOpInterface"> {
  let cppNamespace = "::mlir::afir::ascend::kernelize";
  let description = [{
    Provides Ascend Kernelize semantic information for operations that can
    participate in Kernelize dependency analysis.
  }];
  let methods = [
    InterfaceMethod<
      "Populate Kernelize semantic information.",
      "::mlir::LogicalResult",
      "populateKernelizeSemanticInfo",
      (ins "::mlir::afir::ascend::kernelize::KernelizeOpSemanticInfo &":$info)
    >
  ];
}

#endif
