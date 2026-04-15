#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::runtime {

llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s",
                                   path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

llvm::Expected<std::string> readTextFileOrErr(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return llvm::createStringError(bufferOr.getError(),
                                   "Cannot read file: %s",
                                   path.str().c_str());
  return (*bufferOr)->getBuffer().str();
}

llvm::Error ensureDirectory(llvm::StringRef path) {
  if (auto ec = llvm::sys::fs::create_directories(path))
    return llvm::createStringError(ec, "Cannot create directory: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

llvm::Error ensureFileExists(llvm::StringRef path, llvm::StringRef stage,
                             llvm::StringRef context) {
  if (llvm::sys::fs::exists(path))
    return llvm::Error::success();
  const std::string contextText = context.str();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "[%s] completed but did not create expected file: %s%s%s",
      stage.str().c_str(), path.str().c_str(), context.empty() ? "" : " (",
      context.empty() ? "" : contextText.c_str(),
      context.empty() ? "" : ")");
}

std::string joinPath(llvm::StringRef base, llvm::StringRef leaf) {
  llvm::SmallString<256> joined(base);
  llvm::sys::path::append(joined, leaf);
  return joined.str().str();
}

std::string makeStageContext(
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> fields) {
  std::string out;
  llvm::raw_string_ostream os(out);
  bool first = true;
  for (const auto &field : fields) {
    if (field.second.empty())
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << field.first << "=" << field.second;
  }
  os.flush();
  return out;
}

llvm::Error runProcess(const std::vector<std::string> &args,
                       llvm::StringRef stage, llvm::StringRef context) {
  if (args.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "[%s] received an empty command",
                                   stage.str().c_str());

  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (const auto &arg : args)
    argv.push_back(arg);

  std::string errMsg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      300, 0, &errMsg);
  if (ret == 0)
    return llvm::Error::success();

  const std::string program = argv[0].str();
  const std::string contextText = context.str();
  std::string renderedCommand = renderCommandForDebug(args);
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "[%s] program=%s%s%s failed (exit %d): %s\n  command: %s",
      stage.str().c_str(), program.c_str(), context.empty() ? "" : " inputs: ",
      context.empty() ? "" : contextText.c_str(), ret, errMsg.c_str(),
      renderedCommand.c_str());
}

} // namespace mlir::runtime
