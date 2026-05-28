//===- GatherElementwiseFusion.cpp - Fuse elementwise into gather ---------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This transformation runs at the tensor level during ascend-kernelize. It
// fuses chains of:
//
//   pre-op (e.g. relu linalg.generic)
//     -> gather linalg.generic {gather_dim / embedding_dim}
//       -> post-op (e.g. add linalg.generic)
//
// into a single fused gather generic whose body contains all ops inlined.
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeInternalPasses.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace mlir::ascend {

namespace {

// Returns true if op is a gather (has gather_dim or embedding_dim attr).
static bool isGatherOp(linalg::GenericOp op) {
  return op->hasAttr(ascend::kGatherDimAttr) ||
         op->hasAttr(ascend::kEmbeddingDimAttr);
}

// Returns true if the linalg.generic body contains only arith/constant ops
// and a linalg.yield terminator (no tensor.extract, no linalg.index).
static bool isSimpleElementwise(linalg::GenericOp op) {
  if (isGatherOp(op))
    return false;
  for (auto &bodyOp : op.getBody()->without_terminator()) {
    if (isa<tensor::ExtractOp>(bodyOp) || isa<linalg::IndexOp>(bodyOp))
      return false;
  }
  return true;
}

} // namespace

LogicalResult fuseGatherElementwise(func::FuncOp funcOp) {
  // Collect gather ops first; we'll modify as we go.
  SmallVector<linalg::GenericOp> gatherOps;
  funcOp.walk([&](linalg::GenericOp op) {
    if (isGatherOp(op))
      gatherOps.push_back(op);
  });

  for (linalg::GenericOp gatherOp : gatherOps) {
    // --- Find pre-op ---
    // The gather body has a tensor.extract; find its source tensor.
    // If that tensor is produced by a simple elementwise generic → pre-op.
    tensor::ExtractOp extractOp;
    gatherOp.getBody()->walk([&](tensor::ExtractOp e) {
      if (!extractOp)
        extractOp = e;
    });
    if (!extractOp)
      continue;

    Value capturedTensor = extractOp.getTensor();
    linalg::GenericOp preOp;
    if (auto defOp = capturedTensor.getDefiningOp<linalg::GenericOp>())
      if (isSimpleElementwise(defOp))
        preOp = defOp;

    // --- Find post-op ---
    // The gather op has a single result; if it's used as an input to exactly
    // one simple elementwise generic → post-op.
    Value gatherResult = gatherOp->getResult(0);
    linalg::GenericOp postOp;
    for (Operation *user : gatherResult.getUsers()) {
      if (auto candidate = dyn_cast<linalg::GenericOp>(user)) {
        if (isSimpleElementwise(candidate)) {
          postOp = candidate;
          break;
        }
      }
    }

    if (!preOp && !postOp)
      continue; // nothing to fuse

    // Determine insertion point: must be after all operands are defined.
    // The latest-defined operand among gatherOp and postOp operands (including
    // postOutValue, postExtraIns, etc.) determines where we insert.
    // Simplest approach: insert just before postOp if it exists, else before gatherOp,
    // but after the postOutValue (which may be defined between gatherOp and postOp).
    Operation *insertBefore = postOp ? postOp.getOperation()
                                     : gatherOp.getOperation();
    OpBuilder builder(insertBefore);
    Location loc = gatherOp.getLoc();

    // --- Compute fused ins, maps, iterator types ---
    auto gatherMaps = gatherOp.getIndexingMapsArray();
    auto gatherIterTypes = gatherOp.getIteratorTypesArray();

    SmallVector<Value> fusedIns;
    SmallVector<AffineMap> fusedMaps;

    // gather's indices tensor (ins[0]) and its map (maps[0])
    fusedIns.push_back(gatherOp.getDpsInputOperand(0)->get());
    fusedMaps.push_back(gatherMaps[0]);

    // pre-op extra ins (beyond input[0] which is the data tensor)
    SmallVector<Value> preExtraIns;
    SmallVector<AffineMap> preExtraMaps;
    if (preOp) {
      auto preMaps = preOp.getIndexingMapsArray();
      unsigned numPreIns = (unsigned)preOp.getNumDpsInputs();
      for (unsigned i = 1; i < numPreIns; ++i) {
        preExtraIns.push_back(preOp.getDpsInputOperand(i)->get());
        preExtraMaps.push_back(preMaps[i]);
      }
    }
    fusedIns.append(preExtraIns);
    fusedMaps.append(preExtraMaps);

    // post-op extra ins (exclude the one that is the gather result)
    SmallVector<Value> postExtraIns;
    SmallVector<AffineMap> postExtraMaps;
    unsigned postGatherArgIdx = 0;
    Value postOutValue;
    AffineMap postOutMap;
    if (postOp) {
      auto postMaps = postOp.getIndexingMapsArray();
      unsigned numPostIns = (unsigned)postOp.getNumDpsInputs();
      for (unsigned i = 0; i < numPostIns; ++i) {
        if (postOp.getDpsInputOperand(i)->get() == gatherResult) {
          postGatherArgIdx = i;
          continue;
        }
        postExtraIns.push_back(postOp.getDpsInputOperand(i)->get());
        postExtraMaps.push_back(postMaps[i]);
      }
      fusedIns.append(postExtraIns);
      fusedMaps.append(postExtraMaps);
      postOutValue = postOp.getDpsInitOperand(0)->get();
      postOutMap = postMaps[numPostIns]; // out map (last map)
    } else {
      postOutValue = gatherOp.getDpsInitOperand(0)->get();
      postOutMap = gatherMaps[1]; // gather's out map
    }
    fusedMaps.push_back(postOutMap);

    // Offsets into fused block args:
    //   [0]                                  = indices arg
    //   [1 .. 1+preExtraIns.size()-1]        = pre-op extra ins args
    //   [1+preExtraIns.size() .. ...]        = post-op extra ins args
    //   [fusedIns.size()]                    = out init arg
    unsigned preArgOffset  = 1;
    unsigned postArgOffset = preArgOffset + (unsigned)preExtraIns.size();
    unsigned outArgIdx     = postArgOffset + (unsigned)postExtraIns.size();

    // Determine gatherDim for building extract indices.
    int64_t gatherDim = -1;
    if (auto attr = gatherOp->getAttrOfType<IntegerAttr>(
            ascend::kGatherDimAttr))
      gatherDim = attr.getInt();
    else if (auto attr = gatherOp->getAttrOfType<IntegerAttr>(
                 ascend::kEmbeddingDimAttr))
      gatherDim = attr.getInt();

    // Data source tensor: pre-op's first ins (raw data), or captured tensor.
    Value dataSourceTensor =
        preOp ? preOp.getDpsInputOperand(0)->get() : capturedTensor;

    // --- Body builder lambda ---
    // linalg::GenericOp body builder receives (OpBuilder&, Location, ValueRange args).
    auto bodyBuilder = [&](OpBuilder &b, Location bLoc, ValueRange args) {
      // args layout matches fusedIns + outs element types.
      Value idxArg  = args[0];
      Value outArg  = args[outArgIdx];

      // 1. Emit linalg.index for each iteration dimension.
      unsigned rank = (unsigned)gatherIterTypes.size();
      SmallVector<Value> indexVals(rank);
      for (unsigned d = 0; d < rank; ++d)
        indexVals[d] = b.create<linalg::IndexOp>(bLoc, d);

      // 2. Emit arith.index_cast for the indices block arg.
      //    Find the cast type from gather body.
      Type castType = b.getIndexType();
      for (auto &op : gatherOp.getBody()->without_terminator()) {
        if (auto castOp = dyn_cast<arith::IndexCastOp>(op)) {
          castType = castOp.getType();
          break;
        }
      }
      Value castVal = b.create<arith::IndexCastOp>(bLoc, castType, idxArg);

      // 3. Build extract indices: use castVal at gatherDim, linalg.index elsewhere.
      SmallVector<Value> extractIndices(rank);
      for (unsigned d = 0; d < rank; ++d) {
        if ((int64_t)d == gatherDim)
          extractIndices[d] = castVal;
        else
          extractIndices[d] = indexVals[d];
      }

      // 4. tensor.extract from the data source tensor.
      Value rawExtracted =
          b.create<tensor::ExtractOp>(bLoc, dataSourceTensor, extractIndices);

      // 5. Inline pre-op scalar body ops.
      Value preResult = rawExtracted;
      if (preOp) {
        IRMapping preMap;
        preMap.map(preOp.getBody()->getArgument(0), rawExtracted);
        for (unsigned i = 0; i < (unsigned)preExtraIns.size(); ++i)
          preMap.map(preOp.getBody()->getArgument(i + 1),
                     args[preArgOffset + i]);
        preMap.map(
            preOp.getBody()->getArgument(
                preOp.getBody()->getNumArguments() - 1),
            outArg);
        for (auto &op : preOp.getBody()->without_terminator()) {
          Operation *cloned = b.clone(op, preMap);
          for (auto [orig, rep] :
               llvm::zip(op.getResults(), cloned->getResults()))
            preMap.map(orig, rep);
        }
        auto preYield =
            cast<linalg::YieldOp>(preOp.getBody()->getTerminator());
        preResult = preMap.lookupOrDefault(preYield.getValues()[0]);
      }

      // 6. Inline post-op scalar body ops.
      Value finalResult = preResult;
      if (postOp) {
        IRMapping postMap;
        unsigned numPostIns = (unsigned)postOp.getNumDpsInputs();
        unsigned extraIdx = 0;
        for (unsigned i = 0; i < numPostIns; ++i) {
          if (i == postGatherArgIdx)
            postMap.map(postOp.getBody()->getArgument(i), preResult);
          else
            postMap.map(postOp.getBody()->getArgument(i),
                        args[postArgOffset + extraIdx++]);
        }
        postMap.map(postOp.getBody()->getArgument(numPostIns), outArg);
        for (auto &op : postOp.getBody()->without_terminator()) {
          Operation *cloned = b.clone(op, postMap);
          for (auto [orig, rep] :
               llvm::zip(op.getResults(), cloned->getResults()))
            postMap.map(orig, rep);
        }
        auto postYield =
            cast<linalg::YieldOp>(postOp.getBody()->getTerminator());
        finalResult = postMap.lookupOrDefault(postYield.getValues()[0]);
      }

      b.create<linalg::YieldOp>(bLoc, finalResult);
    };

    // --- Create the fused linalg.generic ---
    auto fusedOp = builder.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{postOutValue.getType()},
        /*inputs=*/fusedIns,
        /*outputs=*/ValueRange{postOutValue},
        /*indexingMaps=*/fusedMaps,
        /*iteratorTypes=*/gatherIterTypes,
        /*bodyBuilder=*/bodyBuilder);

    // Copy gather_dim / embedding_dim attribute.
    if (auto attr = gatherOp->getAttr(ascend::kGatherDimAttr))
      fusedOp->setAttr(ascend::kGatherDimAttr, attr);
    if (auto attr = gatherOp->getAttr(ascend::kEmbeddingDimAttr))
      fusedOp->setAttr(ascend::kEmbeddingDimAttr, attr);

    // --- Replace uses and erase old ops ---
    Value fusedResult = fusedOp->getResult(0);
    if (postOp) {
      postOp->getResult(0).replaceAllUsesWith(fusedResult);
      postOp.erase();
    } else {
      gatherResult.replaceAllUsesWith(fusedResult);
    }
    gatherOp.erase();
    if (preOp)
      preOp.erase();
  }
  return success();
}

} // namespace mlir::ascend
