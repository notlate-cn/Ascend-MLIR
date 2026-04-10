// RUN: not afir-translate -mlir-to-cann %S/cann-translate-mix-island-input.mlir 2>&1 | FileCheck %s

// This fixture adds an unrelated vector-only sink that consumes the bias
// broadcast branch but does not participate in the boundary-rooted execution
// cone. Generic single-chain validation must reject it instead of accepting a
// disconnected vector island.
// CHECK: error: 'func.func' op mix translation requires a supported cube/vector partitioned kernel shape; generic single-chain analysis rejected plan because vector region contains ops outside the single executable chain
