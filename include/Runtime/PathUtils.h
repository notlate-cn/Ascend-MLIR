#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

std::string getHostCannArchDir();
std::string getHostCannArchDir(llvm::StringRef machine);

std::string findFirstExistingPath(llvm::ArrayRef<std::string> candidates);
llvm::Expected<std::string> requireAscendHome();
std::string findAscendHome();
std::string findSocVersion();
std::string resolveSocVersion(llvm::StringRef explicitSocVersion,
                              llvm::StringRef fallbackSocVersion = "");
std::string resolveAscendHomeForTest(llvm::StringRef ascendHomeEnv,
                                     llvm::StringRef toolkitHomeEnv);
std::string resolveSocVersionForTest(llvm::StringRef explicitSocVersion,
                                     llvm::StringRef envSocVersion,
                                     llvm::StringRef fallbackSocVersion);

std::string findAscendIncludeDir(llvm::StringRef ascendHome);
std::string findAscendIncludeDir(llvm::StringRef ascendHome,
                                 llvm::StringRef machine);
std::string findAscendLib64Dir(llvm::StringRef ascendHome);
std::string findAscendLib64Dir(llvm::StringRef ascendHome,
                               llvm::StringRef machine);
std::string findAscendDeviceLibDir(llvm::StringRef ascendHome);
std::string findAscendDeviceLibDir(llvm::StringRef ascendHome,
                                   llvm::StringRef machine);
std::string findAscendSimulatorLibDir(llvm::StringRef ascendHome,
                                      llvm::StringRef socVersion);
std::string findAscendSimulatorLibDir(llvm::StringRef ascendHome,
                                      llvm::StringRef socVersion,
                                      llvm::StringRef machine);
std::string findAscendDavSimulatorLibDir(llvm::StringRef ascendHome);
std::string findAscendDavSimulatorLibDir(llvm::StringRef ascendHome,
                                         llvm::StringRef machine);
llvm::Expected<std::string>
requireAscendDavSimulatorLibDir(llvm::StringRef ascendHome);
llvm::Expected<std::string>
requireAscendDavSimulatorLibDir(llvm::StringRef ascendHome,
                                llvm::StringRef machine);
std::string findAscendAclLibPath(llvm::StringRef ascendHome);
std::string findAscendAclLibPath(llvm::StringRef ascendHome,
                                 llvm::StringRef machine);
std::string findAscendRuntimeCamodelPath(llvm::StringRef ascendHome,
                                         llvm::StringRef socVersion);
std::string findAscendRuntimeCamodelPath(llvm::StringRef ascendHome,
                                         llvm::StringRef socVersion,
                                         llvm::StringRef machine);
std::string findAscendTikcppDir(llvm::StringRef ascendHome);
std::string findAscendAscDir(llvm::StringRef ascendHome);

} // namespace mlir::runtime
