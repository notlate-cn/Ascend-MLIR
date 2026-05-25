//===- AscendKernelPatternTest.cpp - Ascend kernel pattern tests -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Pattern/KernelPattern.h"

#include "Conversion/Ascend/Kernelize/Candidate/CandidateMergeAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/KernelizeFamilyResolver.h"

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

FusionCandidate makeFusionCandidate(unsigned candidateId,
                                    ArrayRef<Operation *> internalOps,
                                    ArrayRef<Operation *> primaryOps) {
  FusionCandidate candidate;
  candidate.candidateId = candidateId;
  candidate.kind = CandidateKind::Fusion;
  candidate.primitive = KernelizePrimitiveKind::ElementwiseChain;
  candidate.internalOps.append(internalOps.begin(), internalOps.end());
  candidate.primaryOps.append(primaryOps.begin(), primaryOps.end());
  candidate.scheduleContract.templateFamilies.push_back("vector");
  candidate.closure.isClosed = true;
  candidate.legal = true;
  candidate.benefitScore = 10;
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

TEST(AscendCandidateMergeAnalyzerTest, IteratesMergedCandidatesToFixpoint) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp first(context, "test.first");
  TestOp second(context, "test.second");
  TestOp third(context, "test.third");
  TestOp fourth(context, "test.fourth");
  TestOp fifth(context, "test.fifth");

  DependencyAnalysisResult deps;
  deps.index.orderedOps.assign({first, second, third, fourth, fifth});
  deps.index.opIds.try_emplace(first, OperationId{0});
  deps.index.opIds.try_emplace(second, OperationId{1});
  deps.index.opIds.try_emplace(third, OperationId{2});
  deps.index.opIds.try_emplace(fourth, OperationId{3});
  deps.index.opIds.try_emplace(fifth, OperationId{4});
  deps.index.consumers[first].push_back(second);
  deps.index.consumers[second].push_back(third);
  deps.index.consumers[third].push_back(fourth);
  deps.index.consumers[fourth].push_back(fifth);
  deps.index.producers[second].push_back(first);
  deps.index.producers[third].push_back(second);
  deps.index.producers[fourth].push_back(third);
  deps.index.producers[fifth].push_back(fourth);

  SmallVector<FusionCandidate> candidates;
  candidates.push_back(makeFusionCandidate(0, {first, second}, {first}));
  candidates.push_back(makeFusionCandidate(1, {second, third}, {second}));
  candidates.push_back(makeFusionCandidate(2, {third, fourth}, {third}));
  candidates.push_back(makeFusionCandidate(3, {fourth, fifth}, {fourth}));

  KernelizeConfig config;
  config.maxPrimaryRolesPerCandidate = 5;
  SmallVector<MergedCandidate> merged =
      CandidateMergeAnalyzer().analyze(candidates, deps, config);

  ASSERT_FALSE(merged.empty());
  auto fullChainIt = llvm::find_if(merged, [&](const MergedCandidate &candidate) {
    return candidate.internalOps.size() == 5 &&
           candidate.internalOps[0] == static_cast<Operation *>(first) &&
           candidate.internalOps[1] == static_cast<Operation *>(second) &&
           candidate.internalOps[2] == static_cast<Operation *>(third) &&
           candidate.internalOps[3] == static_cast<Operation *>(fourth) &&
           candidate.internalOps[4] == static_cast<Operation *>(fifth);
  });
  ASSERT_NE(fullChainIt, merged.end());
  EXPECT_TRUE(fullChainIt->legal);
  EXPECT_EQ(fullChainIt->scheduleContract.templateFamilies.size(), 1u);
  EXPECT_EQ(fullChainIt->scheduleContract.templateFamilies.front(), "vector");
}

TEST(AscendKernelizeFamilyResolverTest, PrefersCubeForCubeVectorFamilyPair) {
  SmallVector<std::string, 2> lhsFamilies{"cube"};
  SmallVector<std::string, 2> rhsFamilies{"vector"};
  SmallVector<KernelizePrimitiveKind, 2> primitives{
      KernelizePrimitiveKind::ElementwiseChain,
      KernelizePrimitiveKind::ConsumerIntoPrimary};

  KernelizeFamilyResolution resolution =
      resolveKernelizeTemplateFamilies(lhsFamilies, rhsFamilies, primitives);

  ASSERT_EQ(resolution.templateFamilies.size(), 1u);
  EXPECT_EQ(resolution.templateFamilies.front(), "cube");
  EXPECT_EQ(resolution.resolverName, "kernelize_trait_resolver");
}

TEST(AscendKernelPatternBuilderTest, ProducesExplicitPlacementEdgesFromAttrs) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp first(context, "test.first");
  TestOp second(context, "test.second");
  TestOp third(context, "test.third");

  static_cast<Operation *>(first)->setAttr(
      kKernelizeMustCoLocateGroupAttr,
      IntegerAttr::get(IntegerType::get(&context, 32), 0));
  static_cast<Operation *>(second)->setAttr(
      kKernelizeMustCoLocateGroupAttr,
      IntegerAttr::get(IntegerType::get(&context, 32), 0));
  static_cast<Operation *>(second)->setAttr(
      kKernelizeMustSeparateGroupAttr,
      IntegerAttr::get(IntegerType::get(&context, 32), 1));
  static_cast<Operation *>(third)->setAttr(
      kKernelizeMustSeparateGroupAttr,
      IntegerAttr::get(IntegerType::get(&context, 32), 1));

  DependencyAnalysisResult deps;
  deps.index.orderedOps.assign({first, second, third});
  deps.index.opIds.try_emplace(first, OperationId{0});
  deps.index.opIds.try_emplace(second, OperationId{1});
  deps.index.opIds.try_emplace(third, OperationId{2});

  SmallVector<FusionCandidate> candidates;
  candidates.push_back(makeFusionCandidate(0, {first}, {first}));
  candidates.push_back(makeFusionCandidate(1, {second}, {second}));
  candidates.push_back(makeFusionCandidate(2, {third}, {third}));

  KernelPatternGraph graph =
      KernelPatternBuilder().build(candidates, {}, {}, deps);

  bool hasCoLocate = llvm::any_of(graph.edges, [](const KernelPatternEdge &edge) {
    return edge.kind == KernelPatternEdgeKind::MustCoLocate &&
           edge.from == 0 && edge.to == 1;
  });
  bool hasSeparate = llvm::any_of(graph.edges, [](const KernelPatternEdge &edge) {
    return edge.kind == KernelPatternEdgeKind::MustSeparate &&
           edge.from == 1 && edge.to == 2;
  });
  EXPECT_TRUE(hasCoLocate);
  EXPECT_TRUE(hasSeparate);
}
