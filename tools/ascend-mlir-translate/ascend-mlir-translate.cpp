//===- ascend-mlir-translate.cpp - Ascend translation driver -----*- C++ -*-===//
//
// Supports -mlir-to-cann translation for CANN-standard AscendC kernel emission.
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> TilingSpaceOut(
    "tiling-space-out",
    cl::desc("Write tiling_space.json to this path"),
    cl::init(""));

static cl::opt<std::string> ArtifactManifestOut(
    "artifact-manifest-out",
    cl::desc("Write artifact_manifest.json to this path"),
    cl::init(""));

static cl::opt<std::string> HostTilingOut(
    "host-tiling-out",
    cl::desc("Write host tiling C ABI source to this path"),
    cl::init(""));

static cl::opt<std::string> CannSoc(
    "cann-soc",
    cl::desc("CANN SoC string used in generated runtime artifacts"),
    cl::init("Ascend910B1"));

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration cannReg(
      "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
      [](Operation *op, raw_ostream &os) {
        CannTranslationOptions options;
        options.tilingSpaceOutPath = TilingSpaceOut;
        options.artifactManifestOutPath = ArtifactManifestOut;
        options.hostTilingOutPath = HostTilingOut;
        options.soc = CannSoc;
        return translateToCannKernel(op, os, options);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<ascendc::AscendCDialect, emitasc::EmitAscDialect>();
        ascendc::registerExternalModels(registry);
        emitasc::registerExternalModels(registry);
      });

  return failed(mlirTranslateMain(argc, argv, "Ascend MLIR translation tool"));
}
