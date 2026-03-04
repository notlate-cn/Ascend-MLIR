#include "CAPI/Passes.h"

#include "Conversion/Passes.h"
#include "Dialect/AFIR/Transforms/Passes.h"

void mlirRegisterAFIRPasses() {
  mlir::afir::registerAFIRPasses();
  mlir::afir::registerAFIRConversionPasses();
}