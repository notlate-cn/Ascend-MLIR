#include "MixDirectCompileInternal.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir::runtime {

llvm::Expected<MixCompileLayout>
buildMixDirectCompileLayout(llvm::StringRef outputDir,
                            llvm::StringRef kernelName) {
  llvm::SmallString<256> outputRoot(outputDir);
  llvm::sys::fs::make_absolute(outputRoot);
  llvm::SmallString<256> workDir(outputRoot);
  llvm::sys::path::append(workDir, "work");
  llvm::SmallString<256> objectDir(outputRoot);
  llvm::sys::path::append(objectDir, "objects");
  llvm::SmallString<256> outDir(outputRoot);
  llvm::sys::path::append(outDir, "out");
  llvm::SmallString<256> outBinDir(outDir);
  llvm::sys::path::append(outBinDir, "bin");
  llvm::SmallString<256> outIncludeDir(outDir);
  llvm::sys::path::append(outIncludeDir, "include", "ascendc_kernels_sim");
  llvm::SmallString<256> mergeDir(workDir);
  llvm::sys::path::append(mergeDir, "merge_obj");
  llvm::SmallString<256> launcherDir(workDir);
  llvm::sys::path::append(launcherDir, "launcher");
  llvm::SmallString<256> stubDir(workDir);
  llvm::sys::path::append(stubDir, "stub");
  llvm::SmallString<256> hostDir(outputRoot);
  llvm::sys::path::append(hostDir, "host_dir");
  llvm::SmallString<256> hostObjectsDir(hostDir);
  llvm::sys::path::append(hostObjectsDir, "objects-Debug", "host_bisheng_obj");
  llvm::SmallString<256> aicMergeDir(workDir);
  llvm::sys::path::append(aicMergeDir, "aic_merge");
  llvm::SmallString<256> aivMergeDir(workDir);
  llvm::sys::path::append(aivMergeDir, "aiv_merge");
  llvm::SmallString<256> preprocessProbeDir(workDir);
  llvm::sys::path::append(preprocessProbeDir, "preprocess_probe");

  for (llvm::StringRef dir : {outputRoot.str(), workDir.str(), objectDir.str(),
                              outDir.str(), outBinDir.str(),
                              outIncludeDir.str(), mergeDir.str(),
                              launcherDir.str(), stubDir.str(), hostDir.str(),
                              hostObjectsDir.str(), aicMergeDir.str(),
                              aivMergeDir.str(), preprocessProbeDir.str()}) {
    if (auto err = ensureDirectory(dir))
      return std::move(err);
  }

  MixCompileLayout layout;
  layout.outputRoot = outputRoot.str().str();
  layout.workDir = workDir.str().str();
  layout.objectDir = objectDir.str().str();
  layout.outDir = outDir.str().str();
  layout.outBinDir = outBinDir.str().str();
  layout.outIncludeDir = outIncludeDir.str().str();
  layout.mergeDir = mergeDir.str().str();
  layout.launcherDir = launcherDir.str().str();
  layout.stubDir = stubDir.str().str();
  layout.hostDir = hostDir.str().str();
  layout.hostObjectsDir = hostObjectsDir.str().str();
  layout.aicMergeDir = aicMergeDir.str().str();
  layout.aivMergeDir = aivMergeDir.str().str();
  layout.aicObj = joinPath(layout.objectDir, kernelName.str() + "_aic.o");
  layout.aivObj = joinPath(layout.objectDir, kernelName.str() + "_aiv.o");
  layout.aicRelocObj =
      joinPath(layout.objectDir, kernelName.str() + "_aic.reloc.o");
  layout.aivRelocObj =
      joinPath(layout.objectDir, kernelName.str() + "_aiv.reloc.o");
  layout.mergedDeviceObj = joinPath(layout.outDir, "device.o");
  layout.manifestPath = joinPath(layout.outDir, "manifest.txt");
  layout.metadataPath = joinPath(layout.outDir, "mix_metadata.json");
  layout.analysisPath = joinPath(layout.workDir, "analysis.txt");
  layout.mergeDeviceObj = joinPath(layout.mergeDir, "device.o");
  layout.hostStubObjectPath = joinPath(layout.stubDir, "host_stub.o");
  layout.kernelSoPath =
      joinPath(layout.outDir, "lib" + kernelName.str() + "_packed.so");
  layout.mixFlagPath = joinPath(layout.mergeDir, "mix_build.flag");
  layout.runnerMainPath = joinPath(layout.workDir, "main.cpp");
  layout.runnerTilingPath =
      joinPath(layout.workDir, kernelName.str() + "_tiling.cpp");
  layout.runnerDataUtilsPath = joinPath(layout.workDir, "data_utils.h");
  layout.runnerBinaryPath = joinPath(layout.outBinDir, "mix_runner");
  layout.tilingArtifactPath = joinPath(layout.outDir, "tiling.bin");
  layout.launchInfoPath = joinPath(layout.outDir, "launch_info.txt");
  layout.preprocessProbeDir = preprocessProbeDir.str().str();
  layout.aicProbeObject =
      joinPath(layout.preprocessProbeDir, kernelName.str() + "_aic_probe.o");
  layout.aivProbeObject =
      joinPath(layout.preprocessProbeDir, kernelName.str() + "_aiv_probe.o");
  return layout;
}

} // namespace mlir::runtime
