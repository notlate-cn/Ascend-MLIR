#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstring>
#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

static constexpr const char *kStageEmitTilingArtifact = "emit tiling artifact";

static llvm::Expected<uint32_t> readBlockDimFromLaunchInfo(llvm::StringRef path) {
  auto textOr = readTextFileOrErr(path);
  if (!textOr)
    return textOr.takeError();

  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(*textOr).split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.starts_with("block_dim="))
      continue;
    uint64_t value = 0;
    if (line.drop_front(strlen("block_dim=")).getAsInteger(10, value))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid block_dim in launch info: %s",
                                     path.str().c_str());
    return static_cast<uint32_t>(value);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Missing block_dim in launch info: %s",
                                 path.str().c_str());
}

} // namespace

llvm::Expected<MixDirectTilingOutputs>
executeMixDirectTilingStage(const MixCompileLayout &layout,
                            llvm::StringRef runtimeKernelName,
                            llvm::StringRef socVersion,
                            const MixAbiMetadata &abi) {
  MixDirectTilingOutputs outputs;
  outputs.tilingArtifactPath = layout.tilingArtifactPath;
  outputs.launchInfoPath = layout.launchInfoPath;

  std::vector<std::string> tilingEmitCmd = buildMixTilingHelperCommand(
      runtimeKernelName, socVersion, abi.inputs[0].shape, abi.inputs[0].dtype,
      abi.inputs[1].shape, abi.inputs[1].dtype, abi.outputs[0].shape,
      abi.outputs[0].dtype,
      abi.inputs.size() > 2 ? std::optional<DType>(abi.inputs[2].dtype)
                            : std::nullopt,
      layout.tilingArtifactPath, layout.launchInfoPath);
  outputs.runnerCompileCommand = renderCommandForDebug(tilingEmitCmd);

  const std::string tilingArtifactContext = makeStageContext({
      {"helper", tilingEmitCmd.front()},
      {"tiling_artifact", layout.tilingArtifactPath},
      {"launch_info", layout.launchInfoPath},
      {"kernel", runtimeKernelName},
      {"soc_version", socVersion},
  });
  {
    MixDirectStageTimer timer("emit_tiling_artifact", outputs.timings);
    if (auto err = runProcess(tilingEmitCmd, kStageEmitTilingArtifact,
                              tilingArtifactContext))
      return std::move(err);
  }
  if (auto err = ensureFileExists(layout.tilingArtifactPath,
                                  kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return std::move(err);
  if (auto err = ensureFileExists(layout.launchInfoPath,
                                  kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return std::move(err);

  auto blockDimOr = readBlockDimFromLaunchInfo(layout.launchInfoPath);
  if (!blockDimOr)
    return blockDimOr.takeError();
  outputs.blockDim = *blockDimOr;
  outputs.tilingEmitCommand = renderCommandForDebug(tilingEmitCmd);
  return outputs;
}

} // namespace mlir::runtime
