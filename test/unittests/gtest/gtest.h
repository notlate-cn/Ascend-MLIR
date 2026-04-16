#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace testing {

using TestFunc = void (*)();

struct TestCase {
  const char *suite;
  const char *name;
  TestFunc func;
};

inline std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int &failureCount() {
  static int failures = 0;
  return failures;
}

inline void addTest(const char *suite, const char *name, TestFunc func) {
  registry().push_back(TestCase{suite, name, func});
}

struct TestRegistrar {
  TestRegistrar(const char *suite, const char *name, TestFunc func) {
    addTest(suite, name, func);
  }
};

inline void InitGoogleTest(int *, char **) {}

template <typename T>
std::string toString(const T &value) {
  std::ostringstream oss;
  if constexpr (std::is_enum_v<T>) {
    using Underlying = std::underlying_type_t<T>;
    oss << static_cast<Underlying>(value);
  } else {
    oss << value;
  }
  return oss.str();
}

inline void recordFailure(const char *file, int line, const char *expr,
                          const std::string &lhs = std::string(),
                          const std::string &rhs = std::string()) {
  ++failureCount();
  std::cerr << file << ":" << line << ": Failure\n";
  std::cerr << "  Expected: " << expr << "\n";
  if (!lhs.empty() || !rhs.empty()) {
    std::cerr << "  Left: " << lhs << "\n";
    std::cerr << "  Right: " << rhs << "\n";
  }
}

inline int RunAllTests() {
  for (const auto &test : registry()) {
    std::cout << "[ RUN      ] " << test.suite << "." << test.name << "\n";
    const int before = failureCount();
    test.func();
    if (failureCount() == before) {
      std::cout << "[       OK ] " << test.suite << "." << test.name << "\n";
    } else {
      std::cout << "[  FAILED  ] " << test.suite << "." << test.name << "\n";
    }
  }
  std::cout << registry().size() << " test(s), " << failureCount()
            << " failure(s)\n";
  return failureCount() == 0 ? 0 : 1;
}

} // namespace testing

#define TEST(SuiteName, TestName)                                              \
  static void SuiteName##_##TestName##_Test();                                  \
  static ::testing::TestRegistrar SuiteName##_##TestName##_registrar(           \
      #SuiteName, #TestName, &SuiteName##_##TestName##_Test);                   \
  static void SuiteName##_##TestName##_Test()

#define EXPECT_TRUE(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ::testing::recordFailure(__FILE__, __LINE__, "EXPECT_TRUE(" #cond ")"); \
    }                                                                          \
  } while (false)

#define EXPECT_FALSE(cond)                                                     \
  do {                                                                         \
    if (cond) {                                                                \
      ::testing::recordFailure(__FILE__, __LINE__, "EXPECT_FALSE(" #cond ")");\
    }                                                                          \
  } while (false)

#define EXPECT_EQ(lhs, rhs)                                                    \
  do {                                                                         \
    const auto _lhs = (lhs);                                                   \
    const auto _rhs = (rhs);                                                   \
    if (!(_lhs == _rhs)) {                                                     \
      ::testing::recordFailure(__FILE__, __LINE__, "EXPECT_EQ(" #lhs ", " #rhs \
                               ")", ::testing::toString(_lhs),                \
                               ::testing::toString(_rhs));                    \
    }                                                                          \
  } while (false)
