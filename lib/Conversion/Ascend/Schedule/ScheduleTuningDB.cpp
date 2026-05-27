//===- ScheduleTuningDB.cpp - Ascend schedule tuning DB ------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleTuningDB.h"

#include "ScheduleCache.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

bool parseKeyValueLine(StringRef line, llvm::StringMap<std::string> &fields) {
  SmallVector<StringRef, 12> tokens;
  line.split(tokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (tokens.empty())
    return false;

  for (StringRef token : llvm::drop_begin(tokens)) {
    std::pair<StringRef, StringRef> kv = token.split('=');
    if (kv.first.empty() || kv.second.empty())
      return false;
    fields[kv.first] = kv.second.str();
  }
  return true;
}

bool hasSchemaOne(const llvm::StringMap<std::string> &fields) {
  auto it = fields.find("schema");
  return it != fields.end() &&
         it->second == std::to_string(kScheduleTuningDBSchemaVersion);
}

bool hasRequiredRecordFields(const llvm::StringMap<std::string> &fields) {
  return fields.contains("target") && fields.contains("policy") &&
         fields.contains("signature") && fields.contains("family") &&
         fields.contains("template") && fields.contains("result") &&
         fields.contains("tile");
}

bool hasRequiredNegativeFields(const llvm::StringMap<std::string> &fields) {
  return fields.contains("target") && fields.contains("policy") &&
         fields.contains("signature") && fields.contains("reason");
}

bool parseOptionalI64(const llvm::StringMap<std::string> &fields,
                      llvm::StringRef key, std::optional<int64_t> &out) {
  auto it = fields.find(key);
  if (it == fields.end() || it->second.empty())
    return true;
  int64_t value = 0;
  if (llvm::StringRef(it->second).getAsInteger(10, value))
    return false;
  out = value;
  return true;
}

bool containsRecord(const ScheduleTuningDatabase &db,
                    const ScheduleTuningRecord &record) {
  return llvm::any_of(db.records, [&](const ScheduleTuningRecord &candidate) {
    return candidate.target == record.target &&
           candidate.policy == record.policy &&
           candidate.signature == record.signature;
  });
}

} // namespace

FailureOr<ScheduleTuningDatabase> loadScheduleTuningDBFile(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();

  ScheduleTuningDatabase db;
  SmallVector<StringRef, 32> lines;
  StringRef(buffer.get()->getBuffer()).split(lines, '\n');
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.trim();
    if (line.empty())
      continue;

    if (line.starts_with("#")) {
      if (line.starts_with("# ascend.schedule.tuning_db")) {
        llvm::StringMap<std::string> fields;
        if (!parseKeyValueLine(line.drop_front(1).trim(), fields) ||
            !hasSchemaOne(fields))
          return failure();
      }
      continue;
    }

    llvm::StringMap<std::string> fields;
    if (!parseKeyValueLine(line, fields) || !hasSchemaOne(fields))
      return failure();

    if (line.starts_with("record ")) {
      if (!hasRequiredRecordFields(fields))
        return failure();
      ScheduleTuningRecord record;
      record.target = fields["target"];
      record.policy = fields["policy"];
      record.signature = fields["signature"];
      record.family = fields["family"];
      record.templateName = fields["template"];
      record.resultShape = fields["result"];
      record.tileShape = fields["tile"];
      if (!parseOptionalI64(fields, "score", record.score) ||
          !parseOptionalI64(fields, "cycle_count", record.cycleCount))
        return failure();
      record.profilePath = fields.lookup("profile");
      record.source = fields.lookup("source");
      if (!containsRecord(db, record))
        db.records.push_back(std::move(record));
      continue;
    }

    if (line.starts_with("negative ")) {
      if (!hasRequiredNegativeFields(fields))
        return failure();
      ScheduleNegativeRecord record;
      record.target = fields["target"];
      record.policy = fields["policy"];
      record.signature = fields["signature"];
      record.reason = fields["reason"];
      if (!parseOptionalI64(fields, "score", record.score))
        return failure();
      record.profilePath = fields.lookup("profile");
      record.source = fields.lookup("source");
      db.negativeRecords.push_back(std::move(record));
      continue;
    }

    return failure();
  }

  return db;
}

LogicalResult writeScheduleTuningDBFile(StringRef path,
                                        const ScheduleTuningDatabase &db) {
  std::error_code ec;
  llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return failure();

  output.os() << "# ascend.schedule.tuning_db schema="
              << kScheduleTuningDBSchemaVersion << "\n";

  SmallVector<ScheduleTuningRecord, 8> records(db.records.begin(),
                                               db.records.end());
  llvm::sort(records, [](const ScheduleTuningRecord &lhs,
                         const ScheduleTuningRecord &rhs) {
    return std::tie(lhs.target, lhs.policy, lhs.signature) <
           std::tie(rhs.target, rhs.policy, rhs.signature);
  });
  for (const ScheduleTuningRecord &record : records) {
    output.os() << "record schema=" << kScheduleTuningDBSchemaVersion
                << " target=" << record.target
                << " policy=" << record.policy
                << " signature=" << record.signature
                << " family=" << record.family
                << " template=" << record.templateName
                << " result=" << record.resultShape
                << " tile=" << record.tileShape;
    if (record.score)
      output.os() << " score=" << *record.score;
    if (record.cycleCount)
      output.os() << " cycle_count=" << *record.cycleCount;
    if (!record.profilePath.empty())
      output.os() << " profile=" << record.profilePath;
    if (!record.source.empty())
      output.os() << " source=" << record.source;
    output.os() << "\n";
  }

  SmallVector<ScheduleNegativeRecord, 4> negatives(db.negativeRecords.begin(),
                                                   db.negativeRecords.end());
  llvm::sort(negatives, [](const ScheduleNegativeRecord &lhs,
                           const ScheduleNegativeRecord &rhs) {
    return std::tie(lhs.target, lhs.policy, lhs.signature, lhs.reason) <
           std::tie(rhs.target, rhs.policy, rhs.signature, rhs.reason);
  });
  for (const ScheduleNegativeRecord &record : negatives) {
    output.os() << "negative schema=" << kScheduleTuningDBSchemaVersion
                << " target=" << record.target
                << " policy=" << record.policy
                << " signature=" << record.signature
                << " reason=" << record.reason;
    if (record.score)
      output.os() << " score=" << *record.score;
    if (!record.profilePath.empty())
      output.os() << " profile=" << record.profilePath;
    if (!record.source.empty())
      output.os() << " source=" << record.source;
    output.os() << "\n";
  }

  output.keep();
  return success();
}

SmallVector<std::string, 8>
collectMatchingTuningSignatures(const ScheduleTuningDatabase &db,
                                StringRef target, StringRef policy) {
  SmallVector<std::string, 8> signatures;
  for (const ScheduleTuningRecord &record : db.records) {
    if (record.target != target || record.policy != policy)
      continue;
    if (!llvm::is_contained(signatures, record.signature))
      signatures.push_back(record.signature);
  }
  return signatures;
}

LogicalResult appendTuningResultRecords(ScheduleTuningDatabase &db,
                                        StringRef target, StringRef policy,
                                        ArrayRef<TuningResultKey> keys) {
  if (target.contains(' ')) {
    llvm::errs() << "TuningDB target field must not contain whitespace: '"
                 << target << "'\n";
    return failure();
  }
  if (policy.contains(' ')) {
    llvm::errs() << "TuningDB policy field must not contain whitespace: '"
                 << policy << "'\n";
    return failure();
  }
  for (const TuningResultKey &key : keys) {
    ScheduleTuningRecord record;
    record.target = target.str();
    record.policy = policy.str();
    record.signature = getTuningResultSignature(key);
    record.family = key.bucket.family;
    record.templateName = key.templateName;
    record.resultShape = serializeScheduleDims(key.bucket.resultShape);
    record.tileShape = serializeScheduleDims(key.tileShape);
    if (!containsRecord(db, record))
      db.records.push_back(std::move(record));
  }
  return success();
}

} // namespace mlir::afir::ascend::schedule
