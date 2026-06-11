// lib/Runtime/NpyIO.cpp
#include "Runtime/NpyIO.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace mlir::runtime {

llvm::Expected<NDArray> LoadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot open: %s", path.c_str());
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not a .npy file: %s", path.c_str());

  uint8_t major, minor;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t hlen = 0;
  if (major == 1) {
    uint16_t h16;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }

  std::string header(hlen, '\0');
  f.read(header.data(), hlen);

  NDArray arr;

  // Parse shape
  {
    std::regex re(R"('shape'\s*:\s*\(([^)]*)\))");
    std::smatch m;
    if (!std::regex_search(header, m, re))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot parse shape: %s", path.c_str());
    std::istringstream ss(m[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(0, tok.find_first_not_of(" \t"));
      tok.erase(tok.find_last_not_of(" \t,") + 1);
      if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
    }
  }

  // Parse fortran_order (column-major data layout flag).
  bool fortranOrder = false;
  {
    std::regex re(R"('fortran_order'\s*:\s*(True|False))");
    std::smatch m;
    if (std::regex_search(header, m, re))
      fortranOrder = (m[1].str() == "True");
  }

  // Parse dtype
  {
    std::regex re(R"('descr'\s*:\s*'([^']+)')");
    std::smatch m;
    if (!std::regex_search(header, m, re))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot parse dtype: %s", path.c_str());
    std::string descr = m[1].str();
    if      (descr == "<f2" || descr == "=f2") arr.dtype = DType::F16;
    else if (descr == "<V2" || descr == "=V2" || descr == "|V2") arr.dtype = DType::BF16; // numpy stores bf16 as void2
    else if (descr == "<f4" || descr == "=f4") arr.dtype = DType::F32;
    else if (descr == "<i1" || descr == "=i1" ||
             descr == "|i1")                   arr.dtype = DType::INT8;
    else if (descr == "<i4" || descr == "=i4") arr.dtype = DType::INT32;
    else if (descr == "<i8" || descr == "=i8") arr.dtype = DType::INT64;
    else return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                        "Unsupported dtype '%s': %s",
                                        descr.c_str(), path.c_str());
  }

  arr.allocate();
  const size_t nbytes = arr.nbytes();

  if (!fortranOrder || arr.shape.size() < 2) {
    f.read(reinterpret_cast<char*>(arr.data),
           static_cast<std::streamsize>(nbytes));
    if (f.gcount() != static_cast<std::streamsize>(nbytes))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Truncated data in: %s", path.c_str());
    return arr;
  }

  // Fortran (column-major) data section: read it raw, then re-pack into the
  // C-contiguous (row-major) layout that the rest of the runtime assumes.
  std::vector<uint8_t> raw(nbytes);
  f.read(reinterpret_cast<char*>(raw.data()),
         static_cast<std::streamsize>(nbytes));
  if (f.gcount() != static_cast<std::streamsize>(nbytes))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Truncated data in: %s", path.c_str());

  const size_t rank = arr.shape.size();
  const size_t elemBytes = dtypeBytes(arr.dtype);
  std::vector<size_t> fStride(rank);
  size_t accF = 1;
  for (size_t i = 0; i < rank; ++i) {
    fStride[i] = accF;
    accF *= static_cast<size_t>(arr.shape[i]);
  }
  const size_t numElems = arr.numElements();
  uint8_t *dst = static_cast<uint8_t*>(arr.data);
  std::vector<size_t> idx(rank, 0);
  for (size_t linear = 0; linear < numElems; ++linear) {
    size_t srcElem = 0;
    for (size_t d = 0; d < rank; ++d)
      srcElem += idx[d] * fStride[d];
    std::memcpy(dst + linear * elemBytes, raw.data() + srcElem * elemBytes,
                elemBytes);
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < static_cast<size_t>(arr.shape[d]))
        break;
      idx[d] = 0;
    }
  }
  return arr;
}

llvm::Error SaveNpy(const std::string& path, const NDArray& arr) {
  std::filesystem::path targetPath(path);
  std::filesystem::path parentPath =
      targetPath.has_parent_path() ? targetPath.parent_path()
                                   : std::filesystem::current_path();

  llvm::SmallString<256> tempPattern(parentPath.string());
  llvm::sys::path::append(
      tempPattern,
      targetPath.filename().string() + ".%%%%%%%%.tmp");

  int tempFd = -1;
  llvm::SmallString<256> tempPath;
  if (auto ec = llvm::sys::fs::createUniqueFile(tempPattern, tempFd, tempPath))
    return llvm::createStringError(ec, "Cannot create temp npy file for: %s",
                                   path.c_str());
  ::close(tempFd);

  std::ofstream f(tempPath.c_str(), std::ios::binary | std::ios::trunc);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write: %s", tempPath.c_str());

  const char* descr = nullptr;
  switch (arr.dtype) {
    case DType::F16:   descr = "<f2"; break;
    case DType::BF16:  descr = "<V2"; break;
    case DType::F32:   descr = "<f4"; break;
    case DType::INT8:  descr = "|i1"; break;
    case DType::INT32: descr = "<i4"; break;
    case DType::INT64: descr = "<i8"; break;
  }
  std::string shape_str = "(";
  for (size_t i = 0; i < arr.shape.size(); ++i) {
    shape_str += std::to_string(arr.shape[i]);
    if (arr.shape.size() == 1 || i + 1 < arr.shape.size()) shape_str += ",";
  }
  shape_str += ")";

  std::string dict = "{'descr': '" + std::string(descr) +
                     "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // Pad to multiple of 64 after 10-byte prefix (magic + ver + hlen)
  size_t dict_len = dict.size() + 1; // +1 for '\n'
  size_t total    = 10 + dict_len;
  size_t pad      = (64 - total % 64) % 64;
  dict.append(pad, ' ');
  dict += '\n';

  uint16_t hlen = static_cast<uint16_t>(dict.size());
  f.write("\x93NUMPY", 6);
  f.put(1); f.put(0);
  f.write(reinterpret_cast<char*>(&hlen), 2);
  f.write(dict.data(), dict.size());
  f.write(reinterpret_cast<const char*>(arr.data), arr.nbytes());
  f.close();

  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot finalize temp npy file: %s",
                                   tempPath.c_str());

  std::error_code removeEc;
  std::filesystem::remove(targetPath, removeEc);

  std::error_code renameEc;
  std::filesystem::rename(tempPath.c_str(), targetPath, renameEc);
  if (renameEc) {
    std::filesystem::remove(tempPath.c_str(), removeEc);
    return llvm::createStringError(renameEc,
                                   "Cannot replace npy output: %s",
                                   path.c_str());
  }
  return llvm::Error::success();
}

} // namespace mlir::runtime
