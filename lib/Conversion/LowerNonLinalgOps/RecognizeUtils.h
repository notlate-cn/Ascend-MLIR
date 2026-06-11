//===- RecognizeUtils.h - shared helpers for the recognize-* passes -------===//
//
// Structural-match primitives shared by the aclnn pattern-folding passes
// (recognize-layernorm, recognize-batchnorm, recognize-embedding).  These walk
// linalg.generic def/use chains to identify a decomposed op's affine operands.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::afir {

// True iff `g`'s body region contains an op of type OpT.
template <typename OpT>
inline bool bodyHas(linalg::GenericOp g) {
  bool found = false;
  g.getBody()->walk([&](OpT) { found = true; });
  return found;
}

// The UNIQUE linalg.generic user of `v` whose body contains an OpT.  Returns
// null if there is no such user OR if there is more than one: with multiple
// candidates (e.g. the gamma-scaled value also feeds a residual/second-bias
// add) we cannot tell which generic is the real affine op, and guessing would
// silently fold with the wrong gamma/beta.  The caller bails on null rather
// than guess.
template <typename OpT>
inline linalg::GenericOp userGenericWithBody(Value v) {
  linalg::GenericOp found;
  for (Operation *u : v.getUsers())
    if (auto g = dyn_cast<linalg::GenericOp>(u))
      if (bodyHas<OpT>(g)) {
        if (found)
          return {}; // ambiguous
        found = g;
      }
  return found;
}

// The (single) other tensor input of a 2-input linalg.generic.
inline Value otherInput(linalg::GenericOp g, Value known) {
  for (Value in : g.getInputs())
    if (in != known)
      return in;
  return {};
}

} // namespace mlir::afir
