//===- afir.cpp - End-to-end AFIR transformation tool -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This tool provides an end-to-end transformation pipeline for AFIR dialect,
// supporting conversions between different IR levels.
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIRDialect.h"
#include "Dialect/AFIR/AFIROps.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

// Command line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> emitAFIR("emit-afir",
                              cl::desc("Emit AFIR dialect output"),
                              cl::init(false));

static cl::opt<bool> emitASCIR("emit-ascir",
                               cl::desc("Emit ASC-IR dialect output"),
                               cl::init(true));

static cl::opt<bool> runShapeInference("shape-inference",
                                       cl::desc("Run shape inference pass"),
                                       cl::init(true));

static cl::opt<bool> runCanonicalize("canonicalize",
                                     cl::desc("Run canonicalization pass"),
                                     cl::init(true));

/// Build the transformation pipeline based on command line options.
static LogicalResult buildPipeline(PassManager &pm) {
  // Run shape inference if requested
  if (runShapeInference)
    pm.addPass(afir::createAFIRShapeInferencePass());

  // Run canonicalization if requested
  if (runCanonicalize)
    pm.addPass(afir::createAFIRCanonicalizePass());

  // Convert to ASC-IR if requested
  if (emitASCIR)
    pm.addPass(afir::createConvertAFIRToASCIRPass());

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register command line options
  cl::ParseCommandLineOptions(argc, argv, "AFIR transformation tool\n");

  // Set up the MLIR context and registry
  DialectRegistry registry;
  registry.insert<afir::AFIRDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Register passes
  afir::registerAFIRPasses();
  afir::registerConversionPasses(); 

  // Read the input file
  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    errs() << errorMessage << "\n";
    return 1;
  }

  // Parse the input
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    errs() << "Error parsing input file\n";
    return 1;
  }

  // Set up the pass manager
  PassManager pm(&context);
  if (failed(buildPipeline(pm))) {
    errs() << "Failed to build pass pipeline\n";
    return 1;
  }

  // Run the passes
  if (failed(pm.run(*module))) {
    errs() << "Pass pipeline failed\n";
    return 1;
  }

  // Write the output
  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    errs() << errorMessage << "\n";
    return 1;
  }

  module->print(outputFile->os());
  outputFile->keep();

  return 0;
}