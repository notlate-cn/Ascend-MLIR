#include "KernelizeExternalModels.h"

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {

void registerKernelizeExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    (void)context;
  });
  registry.addExtension(+[](MLIRContext *context, tensor::TensorDialect *) {
    (void)context;
  });
}

} // namespace mlir::afir::ascend::kernelize
