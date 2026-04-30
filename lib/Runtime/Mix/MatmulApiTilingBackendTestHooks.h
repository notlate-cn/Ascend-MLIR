#pragma once

namespace matmul_tiling {
class MatmulApiTiling;
} // namespace matmul_tiling

namespace optiling {
class TCubeTiling;
} // namespace optiling

namespace mlir::runtime {

using MatmulApiTilingGetTilingHook =
    int (*)(matmul_tiling::MatmulApiTiling &, optiling::TCubeTiling &);

void setMatmulApiTilingGetTilingForTest(MatmulApiTilingGetTilingHook hook);

} // namespace mlir::runtime
