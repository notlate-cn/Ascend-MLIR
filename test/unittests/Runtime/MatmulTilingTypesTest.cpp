#include "Runtime/Mix/MatmulTilingTypes.h"
#include <iostream>

using namespace mlir::runtime;

static int g_pass = 0, g_fail = 0;

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (cond) {                                                        \
      std::cout << "  PASS: " << (msg) << "\n";                       \
      ++g_pass;                                                        \
    } else {                                                           \
      std::cerr << "  FAIL: " << (msg) << "\n";                       \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

static void testDefaultRequest() {
  std::cout << "\n[default request]\n";

  MatmulTilingRequest request;

  EXPECT(request.problem.layoutA == MatmulLayout::ND,
         "layoutA defaults to ND");
  EXPECT(request.problem.layoutB == MatmulLayout::ND,
         "layoutB defaults to ND");
  EXPECT(request.problem.layoutC == MatmulLayout::ND,
         "layoutC defaults to ND");
  EXPECT(request.fusion.epilogue == EpilogueKind::None,
         "epilogue defaults to None");
  EXPECT(!request.hints.preferTraverse.has_value(),
         "preferTraverse defaults to empty");
  EXPECT(request.problem.batchShape.empty(), "batchShape defaults to empty");
  EXPECT(!request.problem.transA, "transA defaults to false");
  EXPECT(!request.problem.transB, "transB defaults to false");
  EXPECT(!request.problem.hasBias, "hasBias defaults to false");
  EXPECT(request.problem.M == 0, "problem.M defaults to 0");
  EXPECT(request.problem.N == 0, "problem.N defaults to 0");
  EXPECT(request.problem.K == 0, "problem.K defaults to 0");
}

int main() {
  testDefaultRequest();
  std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail > 0 ? 1 : 0;
}
