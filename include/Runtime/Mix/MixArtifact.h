// include/Runtime/MixArtifact.h
#pragma once

#include "Runtime/Execution/TaskGraph.h"

#include <string>

namespace mlir::runtime {

struct MixArtifact {
  // Required: populated for every successful direct-backend compile.
  std::string kernel_name;
  std::string soc_version;
  std::string work_dir;
  std::string build_dir;
  std::string install_dir;
  std::string kernel_so_path;
  std::string launcher_header_dir;
  std::string host_runner_path;
  std::string manifest_path;
  std::string metadata_path;
  // Optional: may remain empty until a backend produces a merged device object.
  std::string device_object_path;
  // Optional: may remain empty until a backend emits a host stub source file.
  std::string host_stub_source_path;
};

KernelArtifact normalizeMixArtifact(const MixArtifact &artifact,
                                    KernelKind kind = KernelKind::Mix,
                                    MixResourceType mixResourceType =
                                        MixResourceType::Unknown);

} // namespace mlir::runtime
