#include "Runtime/AclnnBackend.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>

namespace mlir::runtime {

namespace {

// ---------------------------------------------------------------------------
// CoordEmitter: walks the coordinator func and emits the network_impl body.
// ---------------------------------------------------------------------------
class CoordEmitter {
public:
  CoordEmitter(ModuleOp module, llvm::raw_ostream &os,
               const AclnnBackendConfig &cfg)
      : module_(module), os_(os), cfg_(cfg) {}

  void emit(func::FuncOp coord) {
    // Map func arguments to inputs[N].
    for (auto [i, arg] : llvm::enumerate(coord.getArguments()))
      names_[arg] = "inputs[" + std::to_string(i) + "]";

    for (Operation &op : coord.front())
      emitOp(op);
  }

private:
  std::string fresh() { return "t" + std::to_string(nextTmp_++); }

  std::string nameOf(Value v) {
    auto it = names_.find(v);
    if (it != names_.end()) return it->second;
    return "/*unknown*/";
  }

  void emitOp(Operation &op) {
    if (auto callOp = dyn_cast<func::CallOp>(op))
      return emitCall(callOp);
    if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
      names_[castOp.getResult()] = nameOf(castOp.getSource());
      return;
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      emitReshapeView(collapseOp.getSrc(), collapseOp.getResult());
      return;
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      emitReshapeView(expandOp.getSrc(), expandOp.getResult());
      return;
    }
    if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {
      // A DPS init / scratch buffer passed to a kernel as an operand.  It must
      // be allocated (with shape) so the kernel can write it and the camodel
      // npy I/O can stage it; an unallocated TensorInfo crashes SaveNpy.
      std::string n = fresh();
      auto rt = dyn_cast<RankedTensorType>(emptyOp.getType());
      int id = rt ? dtypeIdFor(rt.getElementType()) : -1;
      int eb = rt ? elemBytesFor(rt.getElementType()) : 0;
      if (!rt || !rt.hasStaticShape() || id < 0 || eb == 0) {
        os_ << "  // WARNING: unsupported tensor.empty " << emptyOp.getType()
            << "\n";
        os_ << "  TensorInfo " << n << ";\n";
        names_[emptyOp.getResult()] = n;
        return;
      }
      os_ << "  TensorInfo " << n << "; " << n << ".rank=" << rt.getRank()
          << "; " << n << ".dtype=" << id << "; // tensor.empty\n";
      for (auto [d, sz] : llvm::enumerate(rt.getShape()))
        os_ << "  " << n << ".shape[" << d << "]=" << sz << ";\n";
      os_ << "  mlir::runtime::aclnn::rowMajorStrides(" << n << ".shape, " << n
          << ".rank, " << n << ".strides);\n";
      // DPS-init buffers may be wired into kernel ABI slots that the kernel
      // never reads in its body — but the host still stages the bytes H2D.
      // `::operator new` returns uninitialized memory: glibc keeps free-list
      // pointers in just-freed chunks, so the H2D'd device buffer ends up
      // holding ptr-encoded bytes (0x0000ffff aa..) instead of valid f32.
      // The kernel's unrelated scalar paths can fault on those bytes (real-NPU
      // "GM address accessed by scalar exceeds 48 bits" on BERT group20).
      // Zero-init defends against this without touching kernel codegen.
      os_ << "  { size_t _b = (size_t)" << rt.getNumElements() << "*" << eb
          << "; " << n << ".data = ::operator new(_b); std::memset("
          << n << ".data, 0, _b); }\n";
      names_[emptyOp.getResult()] = n;
      return;
    }
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      emitExtractSlice(sliceOp);
      return;
    }
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      emitPad(padOp);
      return;
    }
    if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
      emitReturn(retOp);
      return;
    }
    if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
      // A tensor constant (weight) used as a kernel arg: bake its raw bytes
      // into a static array and build a host TensorInfo pointing at it.  Scalar
      // constants are rematerialized inside kernels by the outliner, so they
      // never reach here as args — skip them.
      auto rt = dyn_cast<RankedTensorType>(cstOp.getType());
      if (!rt)
        return; // scalar const: rematerialized in kernels
      int id = dtypeIdFor(rt.getElementType());
      int eb = elemBytesFor(rt.getElementType());
      if (id < 0 || eb == 0) {
        os_ << "  // WARNING: unsupported const dtype for "
            << cstOp.getResult().getType() << "\n";
        return;
      }

      // Raw element bytes, from either inline dense or a dense_resource blob
      // (torch-imported weights).  splat is only possible for inline dense.
      llvm::ArrayRef<char> raw;
      bool splat = false;
      if (auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue())) {
        raw = dense.getRawData();
        splat = dense.isSplat();
      } else if (auto resAttr =
                     dyn_cast<DenseResourceElementsAttr>(cstOp.getValue())) {
        if (auto *blob = resAttr.getRawHandle().getBlob())
          raw = blob->getData();
      }
      if (raw.empty()) {
        os_ << "  // WARNING: constant weight has no data (elided?); "
            << cstOp.getResult().getType() << "\n";
        return;
      }

      std::string n = fresh();
      int64_t numEl = rt.getNumElements();
      os_ << "  static const unsigned char " << n << "_data[] = {";
      bool first = true;
      auto emitByte = [&](unsigned char b) {
        if (!first)
          os_ << ",";
        os_ << (unsigned)b;
        first = false;
      };
      if (splat)
        for (int64_t e = 0; e < numEl; ++e)
          for (int b = 0; b < eb; ++b)
            emitByte((unsigned char)raw[b]);
      else
        for (char c : raw)
          emitByte((unsigned char)c);
      os_ << "};\n";
      os_ << "  TensorInfo " << n << "; " << n << ".rank=" << rt.getRank()
          << "; " << n << ".dtype=" << id << ";\n";
      for (auto [d, sz] : llvm::enumerate(rt.getShape()))
        os_ << "  " << n << ".shape[" << d << "]=" << sz << ";\n";
      os_ << "  " << n << ".data=(void*)" << n << "_data;\n";
      os_ << "  mlir::runtime::aclnn::rowMajorStrides(" << n << ".shape, " << n
          << ".rank, " << n << ".strides);\n";
      names_[cstOp.getResult()] = n;
      return;
    }
    // scalar arith.constant / linalg.fill etc. — rematerialized in kernels.
  }

  // Materialize a static tensor.extract_slice as a host-side strided copy into
  // a fresh dense row-major buffer.  The source value is assumed contiguous
  // row-major for the slice's source TYPE (true for the kernel-output /
  // collapse_shape / expand_shape chains the outliner produces); we therefore
  // compute the source strides from the source type at emit time rather than
  // reading srcName.strides (which, through a collapse alias, describe the
  // pre-collapse rank).  Rank-reducing slices (dropped unit dims) fall out for
  // free: iterating the slice sizes row-major and writing dst sequentially
  // yields the dense row-major result.
  void emitExtractSlice(tensor::ExtractSliceOp sliceOp) {
    auto srcType = sliceOp.getSourceType();
    auto resType = cast<RankedTensorType>(sliceOp.getResult().getType());
    auto offsets = sliceOp.getStaticOffsets();
    auto sizes = sliceOp.getStaticSizes();
    auto strides = sliceOp.getStaticStrides();

    auto anyDyn = [](llvm::ArrayRef<int64_t> xs) {
      return llvm::any_of(xs, ShapedType::isDynamic);
    };
    int id = dtypeIdFor(resType.getElementType());
    int eb = elemBytesFor(resType.getElementType());
    if (anyDyn(offsets) || anyDyn(sizes) || anyDyn(strides) || id < 0 ||
        eb == 0) {
      os_ << "  // WARNING: unsupported extract_slice (dynamic or bad dtype) "
          << sliceOp.getResult().getType() << "\n";
      return;
    }

    int64_t srcRank = srcType.getRank();
    auto srcShape = srcType.getShape();
    SmallVector<int64_t> srcStrides(srcRank, 1);
    for (int64_t d = srcRank - 2; d >= 0; --d)
      srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
    int64_t numEl = 1;
    for (int64_t s : sizes)
      numEl *= s;

    std::string src = nameOf(sliceOp.getSource());
    std::string n = fresh();
    auto arr = [&](StringRef name, llvm::ArrayRef<int64_t> xs) {
      os_ << "    const int64_t " << name << "[] = {";
      for (auto [i, x] : llvm::enumerate(xs))
        os_ << (i ? "," : "") << x;
      os_ << "};\n";
    };

    os_ << "  TensorInfo " << n << "; " << n << ".rank=" << resType.getRank()
        << "; " << n << ".dtype=" << id << ";\n";
    for (auto [d, sz] : llvm::enumerate(resType.getShape()))
      os_ << "  " << n << ".shape[" << d << "]=" << sz << ";\n";
    os_ << "  mlir::runtime::aclnn::rowMajorStrides(" << n << ".shape, " << n
        << ".rank, " << n << ".strides);\n";
    os_ << "  { // tensor.extract_slice\n";
    arr("_off", offsets);
    arr("_sz", sizes);
    arr("_st", strides);
    arr("_ss", srcStrides);
    os_ << "    size_t _ne = " << numEl << "; " << n
        << ".data = ::operator new(_ne*" << eb << ");\n";
    os_ << "    int64_t _idx[" << srcRank << "] = {0};\n";
    os_ << "    for (size_t _o = 0; _o < _ne; ++_o) {\n";
    os_ << "      size_t _s = 0; for (int _d = 0; _d < " << srcRank
        << "; ++_d) _s += (size_t)(_off[_d] + _idx[_d]*_st[_d]) * (size_t)_ss[_d];\n";
    os_ << "      memcpy((char*)" << n << ".data + _o*" << eb
        << ", (const char*)" << src << ".data + _s*" << eb << ", " << eb
        << ");\n";
    os_ << "      for (int _d = " << srcRank - 1
        << "; _d >= 0; --_d) { if (++_idx[_d] < _sz[_d]) break; _idx[_d] = 0; }\n";
    os_ << "    }\n  }\n";
    names_[sliceOp.getResult()] = n;
  }

  // Materialize a static tensor.pad as a host-side allocation, zero-fill, and
  // strided copy of the source into the unpadded interior.  Used for conv
  // padding that torch.export emits at the coordinator level (e.g. ResNet's
  // 224->230 stem pad, the inner 3x3 pad-by-1s).  Source strides are computed
  // from the source TYPE shape (row-major contiguous) the same way emitExtractSlice
  // does — consistent with the contiguous-row-major convention upstream.
  void emitPad(tensor::PadOp padOp) {
    auto srcType = padOp.getSourceType();
    auto resType = cast<RankedTensorType>(padOp.getResult().getType());
    auto lows = padOp.getStaticLow();
    auto highs = padOp.getStaticHigh();

    auto anyDyn = [](llvm::ArrayRef<int64_t> xs) {
      return llvm::any_of(xs, ShapedType::isDynamic);
    };
    int id = dtypeIdFor(resType.getElementType());
    int eb = elemBytesFor(resType.getElementType());
    if (anyDyn(lows) || anyDyn(highs) || !srcType.hasStaticShape() ||
        !resType.hasStaticShape() || id < 0 || eb == 0) {
      os_ << "  // WARNING: unsupported tensor.pad (dynamic or bad dtype) "
          << padOp.getResult().getType() << "\n";
      return;
    }

    // Extract the static pad scalar (yielded by the pad region's terminator).
    // Conv-padding always yields a constant 0.0 — that's all we support here;
    // zero memset works uniformly across f16/bf16/f32 (0 bit pattern is +0.0).
    double padValue = 0.0;
    bool padIsZero = true;
    if (auto yieldOp = dyn_cast<tensor::YieldOp>(
            padOp.getBody()->getTerminator())) {
      if (auto cst = yieldOp.getValue().getDefiningOp<arith::ConstantOp>()) {
        if (auto fa = dyn_cast<FloatAttr>(cst.getValue())) {
          padValue = fa.getValueAsDouble();
          padIsZero = (padValue == 0.0);
        }
      }
    }
    if (!padIsZero) {
      os_ << "  // WARNING: non-zero tensor.pad value " << padValue
          << " not supported by host codegen; result will be zero-filled\n";
    }

    int64_t rank = resType.getRank();
    auto srcShape = srcType.getShape();
    auto resShape = resType.getShape();
    SmallVector<int64_t> srcStrides(rank, 1), dstStrides(rank, 1);
    for (int64_t d = rank - 2; d >= 0; --d) {
      srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
      dstStrides[d] = dstStrides[d + 1] * resShape[d + 1];
    }
    int64_t srcNumEl = 1;
    for (int64_t s : srcShape) srcNumEl *= s;
    int64_t dstNumEl = 1;
    for (int64_t s : resShape) dstNumEl *= s;

    std::string src = nameOf(padOp.getSource());
    std::string n = fresh();
    auto arr = [&](StringRef name, llvm::ArrayRef<int64_t> xs) {
      os_ << "    const int64_t " << name << "[] = {";
      for (auto [i, x] : llvm::enumerate(xs))
        os_ << (i ? "," : "") << x;
      os_ << "};\n";
    };

    os_ << "  TensorInfo " << n << "; " << n << ".rank=" << rank << "; " << n
        << ".dtype=" << id << ";\n";
    for (auto [d, sz] : llvm::enumerate(resShape))
      os_ << "  " << n << ".shape[" << d << "]=" << sz << ";\n";
    os_ << "  mlir::runtime::aclnn::rowMajorStrides(" << n << ".shape, " << n
        << ".rank, " << n << ".strides);\n";
    os_ << "  { // tensor.pad\n";
    arr("_low", lows);
    arr("_src_sh", srcShape);
    arr("_src_st", srcStrides);
    arr("_dst_st", dstStrides);
    os_ << "    size_t _dst_ne = " << dstNumEl << "; " << n
        << ".data = ::operator new(_dst_ne*" << eb << ");\n";
    os_ << "    std::memset(" << n << ".data, 0, _dst_ne*" << eb << ");\n";
    os_ << "    int64_t _idx[" << rank << "] = {0};\n";
    os_ << "    for (size_t _o = 0; _o < " << srcNumEl << "; ++_o) {\n";
    os_ << "      size_t _src_off = 0, _dst_off = 0;\n";
    os_ << "      for (int _d = 0; _d < " << rank << "; ++_d) {\n";
    os_ << "        _src_off += (size_t)_idx[_d] * (size_t)_src_st[_d];\n";
    os_ << "        _dst_off += (size_t)(_idx[_d] + _low[_d]) * (size_t)_dst_st[_d];\n";
    os_ << "      }\n";
    os_ << "      memcpy((char*)" << n << ".data + _dst_off*" << eb
        << ", (const char*)" << src << ".data + _src_off*" << eb << ", " << eb
        << ");\n";
    os_ << "      for (int _d = " << rank - 1
        << "; _d >= 0; --_d) { if (++_idx[_d] < _src_sh[_d]) break; _idx[_d] = 0; }\n";
    os_ << "    }\n  }\n";
    names_[padOp.getResult()] = n;
  }

  // collapse_shape / expand_shape: a contiguous reshape.  Emit a new TensorInfo
  // that shares the source data pointer but takes the result type's shape/rank/
  // (row-major) strides — a pure name alias would leave downstream consumers
  // (e.g. the BNSD rank-4 view feeding FlashAttentionScore) seeing the source
  // rank/shape.
  void emitReshapeView(Value src, Value res) {
    auto rt = dyn_cast<RankedTensorType>(res.getType());
    std::string srcName = nameOf(src);
    std::string n = fresh();
    if (!rt || !rt.hasStaticShape()) {
      names_[res] = srcName; // fall back to alias
      return;
    }
    os_ << "  TensorInfo " << n << " = " << srcName << "; " << n
        << ".rank=" << rt.getRank() << "; // reshape view\n";
    for (auto [d, sz] : llvm::enumerate(rt.getShape()))
      os_ << "  " << n << ".shape[" << d << "]=" << sz << ";\n";
    os_ << "  mlir::runtime::aclnn::rowMajorStrides(" << n << ".shape, " << n
        << ".rank, " << n << ".strides);\n";
    names_[res] = n;
  }

  void emitCall(func::CallOp callOp) {
    auto callee = module_.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee) {
      os_ << "  // WARNING: unknown callee " << callOp.getCallee() << "\n";
      return;
    }

    if (auto opAttr = callee->getAttrOfType<StringAttr>("aclnn.op")) {
      // aclnn direct call → emit run_<Op>(...).
      // All results share one TensorInfo (aclnn ops return a single tensor).
      std::string resName = fresh();
      os_ << "  TensorInfo " << resName << ";\n";
      if (opAttr.getValue() == "Transpose") {
        // Transpose carries its permutation as an attribute; run_Transpose needs
        // it (the shapes alone don't determine the axis order).
        auto perm = callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.perm");
        std::string pn = fresh();
        os_ << "  static const int64_t " << pn << "[] = {";
        for (int i = 0; i < (int)perm.size(); ++i)
          os_ << (i ? "," : "") << perm[i];
        os_ << "};\n";
        os_ << "  run_Transpose(" << nameOf(callOp.getOperand(0)) << ", " << pn
            << ", " << perm.size() << ", &" << resName << ", stream);\n";
      } else if (opAttr.getValue() == "MaxPool2D" ||
                 opAttr.getValue() == "SumPool2D") {
        // 2D pooling.  Operands: (input, window_template, init) per linalg's
        // pooling op convention.  We only need `input` at runtime; the
        // window template is shape-only and the init is the DPS buffer
        // (allocator's responsibility).  Strides, dilations, kernel_size
        // come from attrs stamped by GroupOutline.
        auto strides = callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.strides");
        auto dilations =
            callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.dilations");
        auto ksize =
            callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.kernel_size");
        std::string sn = fresh(), dn = fresh(), kn = fresh();
        os_ << "  static const int64_t " << sn << "[] = {" << strides[0] << ","
            << strides[1] << "};\n";
        os_ << "  static const int64_t " << dn << "[] = {" << dilations[0]
            << "," << dilations[1] << "};\n";
        os_ << "  static const int64_t " << kn << "[] = {" << ksize[0] << ","
            << ksize[1] << "};\n";
        os_ << "  run_" << opAttr.getValue() << "("
            << nameOf(callOp.getOperand(0)) << ", " << kn << ", " << sn << ", "
            << dn << ", &" << resName << ", stream);\n";
      } else if (opAttr.getValue() == "Conv2D") {
        // Conv2D carries strides + dilations as attributes; padding is
        // already materialized in the coordinator via tensor.pad.  Operand
        // order matches GroupOutline boundaryIn: (input, weight, init).
        auto strides = callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.strides");
        auto dilations =
            callee->getAttrOfType<DenseI64ArrayAttr>("aclnn.dilations");
        std::string sn = fresh(), dn = fresh();
        os_ << "  static const int64_t " << sn << "[] = {" << strides[0] << ","
            << strides[1] << "};\n";
        os_ << "  static const int64_t " << dn << "[] = {" << dilations[0]
            << "," << dilations[1] << "};\n";
        os_ << "  run_Conv2D(" << nameOf(callOp.getOperand(0)) << ", "
            << nameOf(callOp.getOperand(1)) << ", "
            << nameOf(callOp.getOperand(2)) << ", " << sn << ", " << dn
            << ", &" << resName << ", stream);\n";
      } else {
        os_ << "  run_" << opAttr.getValue() << "(";
        for (auto [i, arg] : llvm::enumerate(callOp.getOperands())) {
          if (i) os_ << ", ";
          os_ << nameOf(arg);
        }
        os_ << ", &" << resName << ", stream);\n";
      }
      for (auto res : callOp.getResults())
        names_[res] = resName;
      return;
    }

    // AscendC kernel group → call the host launch helper.
    emitAscendCLaunch(callOp);
  }

  // Map MLIR element type → aclDataType numeric id used by TensorInfo.dtype.
  // f16 -> 1 (ACL_FLOAT16), bf16 -> 27, f32 -> 0.  Returns -1 if unknown.
  static int dtypeIdFor(Type elemTy) {
    if (elemTy.isF16()) return 1;
    if (elemTy.isBF16()) return 27;
    if (elemTy.isF32()) return 0;
    return -1;
  }

  // Element size in bytes for the supported dtypes.
  static int elemBytesFor(Type elemTy) {
    if (elemTy.isF16() || elemTy.isBF16()) return 2;
    if (elemTy.isF32()) return 4;
    return 0;
  }

  void emitAscendCLaunch(func::CallOp callOp) {
    // The MLIR call uses the family id (network.mlir-level kernel name,
    // matching network.json's kernels[].id). The actual emitted kernel
    // binary is the variant the autotuner picked (or v0 by default).
    // P1b: cfg_.variantOverrides (populated from tilings JSON keys at
    // generate() time) supplies the kid→variant mapping; absent entry
    // falls back to "<kid>__v0".
    const auto familyId = callOp.getCallee().str();
    auto ovIt = cfg_.variantOverrides.find(familyId);
    const auto kernelName = ovIt != cfg_.variantOverrides.end()
                                ? ovIt->getValue()
                                : familyId + "__v0";
    const int numIn = static_cast<int>(callOp.getNumOperands());
    const int numOut = static_cast<int>(callOp.getNumResults());

    std::string insName = fresh();
    std::string outsName = fresh();

    // Build input TensorInfo array first; outputs may need its shapes for
    // dynamic dims.
    os_ << "  TensorInfo " << insName << "[" << std::max(numIn, 1) << "] = {";
    for (auto [i, arg] : llvm::enumerate(callOp.getOperands())) {
      if (i) os_ << ", ";
      os_ << nameOf(arg);
    }
    os_ << "};\n";

    // Preallocate output TensorInfos. For dynamic dims (kDynamic in the IR),
    // copy the dim from the kernel's first input at runtime — works for the v1
    // elementwise kernels where output shape == first-input shape.
    os_ << "  TensorInfo " << outsName << "[" << std::max(numOut, 1) << "] = {};\n";
    for (auto [ri, res] : llvm::enumerate(callOp.getResults())) {
      auto rtt = dyn_cast<RankedTensorType>(res.getType());
      if (!rtt) {
        os_ << "  // WARNING: non-ranked-tensor result for " << kernelName << "\n";
        continue;
      }
      auto shape = rtt.getShape();
      int dtypeId = dtypeIdFor(rtt.getElementType());
      int elemBytes = elemBytesFor(rtt.getElementType());
      if (dtypeId == -1 || elemBytes == 0) {
        module_.emitError()
            << "AclnnBackend: unsupported element type '"
            << rtt.getElementType() << "' for result " << ri << " of '"
            << kernelName << "'; cannot emit TensorInfo";
        os_ << "#error \"unsupported element type for " << kernelName
            << " result " << ri << "\"\n";
        continue;
      }
      os_ << "  " << outsName << "[" << ri << "].rank = " << shape.size() << ";\n";
      for (auto [di, d] : llvm::enumerate(shape)) {
        if (mlir::ShapedType::isDynamic(d)) {
          os_ << "  " << outsName << "[" << ri << "].shape[" << di << "] = "
              << insName << "[0].shape[" << di << "];\n";
        } else {
          os_ << "  " << outsName << "[" << ri << "].shape[" << di << "] = "
              << d << ";\n";
        }
      }
      os_ << "  " << outsName << "[" << ri << "].dtype = " << dtypeId << ";\n";
      // Compute byte size at runtime so dynamic dims work.
      // Zero-init output buffers too: kernel may leave tail/padding bytes
      // untouched, and any leftover heap garbage would later be staged H2D
      // when the next kernel consumes this buffer as input (see tensor.empty
      // path above for the BERT group20 motivation).
      os_ << "  { size_t _n = " << elemBytes << "; for (int _d = 0; _d < "
          << outsName << "[" << ri << "].rank; ++_d) _n *= (size_t)"
          << outsName << "[" << ri << "].shape[_d]; "
          << outsName << "[" << ri << "].data = ::operator new(_n); "
          << "std::memset(" << outsName << "[" << ri << "].data, 0, _n); }\n";
    }

    // Call the helper.
    os_ << "  if (mlir::runtime::hostLaunchAscendCKernel(\n";
    os_ << "        \"" << kernelName << "\",\n";
    os_ << "        /*kernelBinariesDir=*/\"" << cfg_.kernelBinariesDir
        << "\",\n";
    os_ << "        /*tilingsPath=*/\"" << cfg_.tilingsPath << "\",\n";
    os_ << "        " << insName << ", " << numIn << ", "
        << outsName << ", " << numOut << ") != 0) {\n";
    os_ << "    fprintf(stderr, \"hostLaunchAscendCKernel(" << kernelName
        << ") failed\\n\");\n";
    os_ << "    return;\n";
    os_ << "  }\n";

    // Register result names.
    for (auto [ri, res] : llvm::enumerate(callOp.getResults()))
      names_[res] = outsName + "[" + std::to_string(ri) + "]";
  }

  void emitReturn(func::ReturnOp retOp) {
    for (auto [i, v] : llvm::enumerate(retOp.getOperands()))
      os_ << "  outputs[" << i << "] = " << nameOf(v) << ";\n";
  }

  ModuleOp module_;
  llvm::raw_ostream &os_;
  const AclnnBackendConfig &cfg_;
  llvm::DenseMap<Value, std::string> names_;
  int nextTmp_ = 0;
};

// ---------------------------------------------------------------------------
// Identify the coordinator function: the non-private func in the module.
// ---------------------------------------------------------------------------
static func::FuncOp findCoordinator(ModuleOp module) {
  func::FuncOp coord;
  module.walk([&](func::FuncOp f) {
    if (!f.isPrivate()) {
      coord = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return coord;
}

// ---------------------------------------------------------------------------
// Collect unique aclnn.op values referenced in the coordinator.
// ---------------------------------------------------------------------------
static void collectAclnnOps(ModuleOp module, func::FuncOp coord,
                             llvm::SmallVector<std::string> &ops) {
  llvm::StringSet<> seen;
  for (Operation &op : coord.front()) {
    auto callOp = dyn_cast<func::CallOp>(op);
    if (!callOp) continue;
    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee) continue;
    auto opAttr = callee->getAttrOfType<StringAttr>("aclnn.op");
    if (!opAttr) continue;
    auto name = opAttr.getValue().str();
    if (seen.insert(name).second)
      ops.push_back(name);
  }
}

// ---------------------------------------------------------------------------
// Build the complete network_host.cpp source string.
// ---------------------------------------------------------------------------
static std::string buildNetworkHostCpp(ModuleOp module,
                                       const AclnnBackendConfig &cfg) {
  std::string src;
  llvm::raw_string_ostream os(src);

  func::FuncOp coord = findCoordinator(module);
  if (!coord) {
    os << "// ERROR: no coordinator function found in network.mlir\n";
    return src;
  }

  llvm::SmallVector<std::string> aclnnOps;
  collectAclnnOps(module, coord, aclnnOps);

  // ① File header + includes
  os << "// Auto-generated by AclnnBackend. DO NOT EDIT.\n";
  if (!cfg.tilingsPath.empty())
    os << "// tilings_path: " << cfg.tilingsPath << "\n";
  if (!cfg.kernelBinariesDir.empty())
    os << "// kernel_binaries_dir: " << cfg.kernelBinariesDir << "\n";
  os << "#include \"acl/acl.h\"\n";
  os << "#include \"Runtime/AclnnOps.h\"\n";
  os << "#include \"Runtime/Execution/HostLaunchHelper.h\"\n";
  os << "#include <cstdint>\n";
  os << "#include <cstdio>\n";
  os << "#include <cstring>\n";
  os << "#include <cstdlib>\n";
  os << "\n";
  os << "using TensorInfo = mlir::runtime::aclnn::TensorInfo;\n";
  // Bring all aclnn CPU-reference ops (run_Matmul, run_FlashAttentionScore, ...)
  // into scope so CoordEmitter's unqualified run_<Op>(...) calls resolve.
  os << "using namespace mlir::runtime::aclnn;\n";
  os << "\n";

  // ③ network_impl: coordinator body translated to C++
  (void)coord.getNumArguments();      // counted by CoordEmitter via block args
  (void)coord.getResultTypes().size();
  os << "static void network_impl(\n";
  os << "    TensorInfo inputs[], int /*numInputs*/,\n";
  os << "    TensorInfo outputs[], int /*numOutputs*/,\n";
  os << "    aclrtStream stream) {\n";

  {
    CoordEmitter emitter(module, os, cfg);
    emitter.emit(coord);
  }

  os << "}\n\n";

  // ④ Public entry point — extern "C" so harness.cpp can declare it without mangling
  os << "extern \"C\" void network(\n";
  os << "    TensorInfo inputs[], int numInputs,\n";
  os << "    TensorInfo outputs[], int numOutputs,\n";
  os << "    aclrtStream stream) {\n";
  os << "  // Try aclInit; if it fails (e.g. running on CPU sim with no NPU\n";
  os << "  // device available), fall through in host-mode so aclnn ops dispatch\n";
  os << "  // to AclnnOps.cpp's CPU-reference implementations.\n";
  os << "  static bool initialized = false;\n";
  os << "  if (!initialized) {\n";
  os << "    initialized = true;\n";
  os << "    const char *force = std::getenv(\"ASCEND_MLIR_FORCE_HOST_MODE\");\n";
  os << "    if (force && force[0] && force[0] != '0') {\n";
  os << "      mlir::runtime::aclnn::setHostMode(true);\n";
  os << "    } else {\n";
  os << "      int rc = aclInit(nullptr);\n";
  os << "      if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {\n";
  os << "        mlir::runtime::aclnn::setHostMode(true);\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  network_impl(inputs, numInputs, outputs, numOutputs, stream);\n";
  os << "}\n";

  os << "\nextern \"C\" void network_set_dump_dir(const char *dir) {\n";
  os << "  mlir::runtime::hostLaunchSetDumpIntermediatesDir(dir);\n";
  os << "}\n";

  os << "\nextern \"C\" void network_set_profile_dir(const char *dir) {\n";
  os << "  mlir::runtime::hostLaunchSetProfileDir(dir);\n";
  os << "}\n";

  return src;
}

} // namespace

// ---------------------------------------------------------------------------
// AclnnBackend::generate
// ---------------------------------------------------------------------------
llvm::Error AclnnBackend::generate(const AclnnBackendConfig &cfg) {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect,
                  tensor::TensorDialect,
                  arith::ArithDialect,
                  linalg::LinalgDialect>();

  auto moduleRef = parseSourceFile<ModuleOp>(cfg.networkMlirPath, &ctx);
  if (!moduleRef)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: failed to parse %s",
                                   cfg.networkMlirPath.c_str());

  // Build family→variant overrides from tilings JSON keys: every key shaped
  // "<family>__v<idx>" pins the family's hostLaunch to that variant name.
  // Default (no key, or key without __v) falls through to <family>__v0.
  AclnnBackendConfig effectiveCfg = cfg;
  if (!cfg.tilingsPath.empty()) {
    if (auto bufOr = llvm::MemoryBuffer::getFile(cfg.tilingsPath,
                                                  /*IsText=*/true)) {
      if (auto parsed = llvm::json::parse((*bufOr)->getBuffer())) {
        if (auto *obj = parsed->getAsObject()) {
          for (auto &kv : *obj) {
            llvm::StringRef name = kv.first;
            auto pos = name.rfind("__v");
            if (pos == llvm::StringRef::npos) continue;
            llvm::StringRef tail = name.substr(pos + 3);
            bool allDigit = !tail.empty();
            for (char c : tail) if (!llvm::isDigit(c)) { allDigit = false; break; }
            if (!allDigit) continue;
            effectiveCfg.variantOverrides[name.substr(0, pos)] = name.str();
          }
        }
      } else {
        llvm::consumeError(parsed.takeError());
      }
    }
  }

  std::string cppSrc = buildNetworkHostCpp(*moduleRef, effectiveCfg);

  std::ofstream out(cfg.outputCppPath, std::ios::binary);
  if (!out)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: cannot write %s",
                                   cfg.outputCppPath.c_str());
  out << cppSrc;
  if (!out)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AclnnBackend: write failed for %s",
                                   cfg.outputCppPath.c_str());

  return llvm::Error::success();
}

} // namespace mlir::runtime
