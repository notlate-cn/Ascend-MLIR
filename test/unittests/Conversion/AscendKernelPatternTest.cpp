//===- AscendKernelPatternTest.cpp - Ascend kernel pattern tests -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelPattern.h"

#include "gtest/gtest.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::afir::ascend::kernelize;

namespace {

class TestOp {
public:
  TestOp(MLIRContext &context, StringRef name) {
    OperationState state(UnknownLoc::get(&context), name);
    op = Operation::create(state);
  }

  ~TestOp() {
    if (op)
      op->destroy();
  }

  operator Operation *() const { return op; }

private:
  Operation *op = nullptr;
};

KernelPatternCandidate makeSingleOpCandidate(unsigned candidateId,
                                             Operation *op,
                                             int64_t benefitScore) {
  KernelPatternCandidate candidate;
  candidate.candidateId = candidateId;
  candidate.internalOps.push_back(op);
  candidate.primaryOps.push_back(op);
  candidate.scheduleContract.templateFamilies.push_back("vector");
  candidate.benefitScore = benefitScore;
  return candidate;
}

} // namespace

TEST(AscendKernelPatternEdgeKeyTest, KeepsLargeNodeIdsDistinct) {
  llvm::DenseSet<KernelPatternEdgeKey> edges;

  KernelPatternEdgeKey dataFromOneToZero{
      1, 0, KernelPatternEdgeKind::DataDependency};
  KernelPatternEdgeKey dataFromZeroToLarge{
      0, 1u << 24, KernelPatternEdgeKind::DataDependency};

  EXPECT_TRUE(edges.insert(dataFromOneToZero).second);
  EXPECT_TRUE(edges.insert(dataFromZeroToLarge).second);
  EXPECT_EQ(edges.size(), 2u);
}

TEST(AscendKernelPartitionerTest, MustCoLocateEdgesForceOnePattern) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp first(context, "test.first");
  TestOp second(context, "test.second");

  DependencyAnalysisResult deps;
  deps.index.orderedOps.assign({first, second});
  deps.index.opIds.try_emplace(first, OperationId{0});
  deps.index.opIds.try_emplace(second, OperationId{1});

  KernelPatternGraph graph;
  graph.nodes.push_back(makeSingleOpCandidate(0, first, 10));
  graph.nodes.push_back(makeSingleOpCandidate(1, second, 9));
  graph.edges.push_back(
      KernelPatternEdge{0, 1, KernelPatternEdgeKind::MustCoLocate, Value()});

  SmallVector<KernelPattern> patterns =
      KernelPartitioner().partition(graph, deps);

  ASSERT_EQ(patterns.size(), 1u);
  EXPECT_EQ(patterns[0].internalOps.size(), 2u);
  EXPECT_EQ(patterns[0].primaryOps.size(), 2u);
  EXPECT_EQ(patterns[0].internalOps[0], static_cast<Operation *>(first));
  EXPECT_EQ(patterns[0].internalOps[1], static_cast<Operation *>(second));
}

TEST(AscendKernelPartitionerTest, MustSeparateEdgesRejectConflictingCandidate) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp first(context, "test.first");
  TestOp second(context, "test.second");

  DependencyAnalysisResult deps;
  deps.index.orderedOps.assign({first, second});
  deps.index.opIds.try_emplace(first, OperationId{0});
  deps.index.opIds.try_emplace(second, OperationId{1});

  KernelPatternGraph graph;
  graph.nodes.push_back(makeSingleOpCandidate(0, first, 10));
  graph.nodes.push_back(makeSingleOpCandidate(1, second, 9));
  graph.edges.push_back(
      KernelPatternEdge{0, 1, KernelPatternEdgeKind::MustSeparate, Value()});

  SmallVector<KernelPattern> patterns =
      KernelPartitioner().partition(graph, deps);

  ASSERT_EQ(patterns.size(), 1u);
  EXPECT_EQ(patterns[0].internalOps.size(), 1u);
  EXPECT_EQ(patterns[0].internalOps[0], static_cast<Operation *>(first));
}
