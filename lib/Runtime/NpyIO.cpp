// lib/Runtime/NpyIO.cpp
#include "Runtime/NpyIO.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

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

  // Parse dtype
  {
    std::regex re(R"('descr'\s*:\s*'([^']+)')");
    std::smatch m;
    if (!std::regex_search(header, m, re))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot parse dtype: %s", path.c_str());
    std::string descr = m[1].str();
    if (descr == "<f2" || descr == "=f2") arr.dtype = DType::F16;
    else if (descr == "<f4" || descr == "=f4") arr.dtype = DType::F32;
    else if (descr == "<i4" || descr == "=i4") arr.dtype = DType::INT32;
    else return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                        "Unsupported dtype '%s': %s",
                                        descr.c_str(), path.c_str());
  }

  size_t nbytes = arr.nbytes();
  arr.data = new uint8_t[nbytes];
  f.read(reinterpret_cast<char*>(arr.data),
         static_cast<std::streamsize>(nbytes));
  if (f.gcount() != static_cast<std::streamsize>(nbytes)) {
    delete[] static_cast<uint8_t*>(arr.data);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Truncated data in: %s", path.c_str());
  }
  return arr;
}

llvm::Error SaveNpy(const std::string& path, const NDArray& arr) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write: %s", path.c_str());

  const char* descr = nullptr;
  switch (arr.dtype) {
    case DType::F16:   descr = "<f2"; break;
    case DType::F32:   descr = "<f4"; break;
    case DType::INT32: descr = "<i4"; break;
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
  return llvm::Error::success();
}

} // namespace mlir::runtime
