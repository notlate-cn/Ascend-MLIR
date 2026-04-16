#pragma once

#include "Runtime/Support/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class MatmulLayout { ND, NZ };
enum class EpilogueKind { None, BiasAdd, BiasAddRelu, BiasAddLeakyRelu };
enum class MatrixTraverseKind { FirstM, FirstN };

struct MatmulProblemDesc {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  std::vector<int64_t> batchShape;
  DType dtypeA = DType::F32;
  DType dtypeB = DType::F32;
  DType dtypeC = DType::F32;
  MatmulLayout layoutA = MatmulLayout::ND;
  MatmulLayout layoutB = MatmulLayout::ND;
  MatmulLayout layoutC = MatmulLayout::ND;
  bool transA = false;
  bool transB = false;
  bool hasBias = false;
};

struct MatmulFusionDesc {
  EpilogueKind epilogue = EpilogueKind::None;
  bool preferFuseVectorEpilogue = false;
  uint64_t consumerAlignmentBytes = 0;
};

struct MatmulScheduleHint {
  std::string socVersion;
  std::optional<int64_t> preferBlockDim;
  std::optional<bool> preferSplitK;
  std::optional<MatrixTraverseKind> preferTraverse;
};

struct MatmulTilingRequest {
  std::string kernelName;
  MatmulProblemDesc problem;
  MatmulFusionDesc fusion;
  MatmulScheduleHint hints;
};

struct MatmulTilingResult {
  std::string backendKind;
  std::string strategyName;
  uint32_t blockDim = 0;
  std::vector<uint8_t> tilingData;
  std::optional<int64_t> tileM;
  std::optional<int64_t> tileN;
  std::optional<int64_t> tileK;
  std::string debugNote;
};

} // namespace mlir::runtime
