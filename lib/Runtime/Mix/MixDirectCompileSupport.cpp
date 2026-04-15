#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>
#include <future>
#include <initializer_list>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::runtime {

MixDirectStageTimer::MixDirectStageTimer(
    llvm::StringRef name, std::vector<MixDirectTimingEntry> &entries)
    : name(name.str()), entries(entries),
      start(std::chrono::steady_clock::now()) {}

MixDirectStageTimer::~MixDirectStageTimer() {
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  entries.push_back(
      {name, static_cast<uint64_t>(elapsed < 0 ? 0 : elapsed)});
}

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

llvm::Error
runProcessesInParallel(llvm::ArrayRef<MixDirectProcessCommand> commands) {
  std::vector<std::future<llvm::Error>> futures;
  futures.reserve(commands.size());
  for (const MixDirectProcessCommand &command : commands) {
    futures.push_back(std::async(std::launch::async, [command]() {
      return runProcess(command.args, command.stage, command.context);
    }));
  }

  llvm::Error joined = llvm::Error::success();
  for (auto &future : futures)
    joined = llvm::joinErrors(std::move(joined), future.get());
  return joined;
}

llvm::Error
runProcessesInParallelForTest(
    llvm::ArrayRef<std::tuple<std::vector<std::string>, std::string,
                              std::string>>
        commands) {
  std::vector<MixDirectProcessCommand> processCommands;
  processCommands.reserve(commands.size());
  for (const auto &command : commands) {
    processCommands.push_back(
        {std::get<0>(command), std::get<1>(command), std::get<2>(command)});
  }
  return runProcessesInParallel(processCommands);
}

llvm::Expected<std::string>
serializeMixDirectTimingJson(llvm::ArrayRef<MixDirectTimingEntry> entries) {
  llvm::json::Object root;
  root["schema_version"] = 1;
  uint64_t totalUs = 0;
  llvm::json::Array stages;
  for (const MixDirectTimingEntry &entry : entries) {
    totalUs += entry.elapsedUs;
    llvm::json::Object stage;
    stage["name"] = entry.name;
    stage["elapsed_us"] = static_cast<int64_t>(entry.elapsedUs);
    stages.push_back(std::move(stage));
  }
  root["total_elapsed_us"] = static_cast<int64_t>(totalUs);
  root["stages"] = std::move(stages);

  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os.flush();
  out.push_back('\n');
  return out;
}

llvm::Error
writeMixDirectTimingFile(llvm::StringRef path,
                         llvm::ArrayRef<MixDirectTimingEntry> entries) {
  auto jsonOr = serializeMixDirectTimingJson(entries);
  if (!jsonOr)
    return jsonOr.takeError();
  return writeTextFile(path, *jsonOr);
}

llvm::Expected<std::string>
serializeMixDirectTimingForTest(
    llvm::ArrayRef<std::tuple<std::string, uint64_t>> entries) {
  std::vector<MixDirectTimingEntry> timingEntries;
  timingEntries.reserve(entries.size());
  for (const auto &entry : entries)
    timingEntries.push_back({std::get<0>(entry), std::get<1>(entry)});
  return serializeMixDirectTimingJson(timingEntries);
}

} // namespace mlir::runtime
