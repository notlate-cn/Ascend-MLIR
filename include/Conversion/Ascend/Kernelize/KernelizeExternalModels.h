#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_EXTERNAL_MODELS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_EXTERNAL_MODELS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::ascend::kernelize {

void registerKernelizeExternalModels(DialectRegistry &registry);

} // namespace mlir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_EXTERNAL_MODELS_H
