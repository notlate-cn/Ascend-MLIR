//===- afir-translate.cpp - AFIR translation driver ---------------*- C++ -*-===//
//
// Supports -mlir-to-cann translation for CANN-standard kernel emission.
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration cannReg(
      "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
      [](Operation *op, raw_ostream &os) {
        return translateToCannKernel(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, ascendc::AscendCDialect,
                        emitasc::EmitAscDialect, emitc::EmitCDialect,
                        func::FuncDialect, LLVM::LLVMDialect, math::MathDialect,
                        memref::MemRefDialect, scf::SCFDialect>();
        ascendc::registerExternalModels(registry);
        emitasc::registerExternalModels(registry);
      });

  return failed(mlirTranslateMain(argc, argv, "AFIR translation tool"));
}
