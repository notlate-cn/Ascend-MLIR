//===- Elementwise.h - AFIR Elementwise ops to ASC-IR conversion -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the conversion patterns for AFIR elementwise operations
// to ASC-IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
#define MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class TypeConverter;

namespace afir {

void populateLoweringAFIRElementwiseOpToASCIRPattern(RewritePatternSet &patterns, MLIRContext *ctx,
                                                     TypeConverter &typeConverter);

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
