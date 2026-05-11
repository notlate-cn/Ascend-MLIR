//===- AscendTargetIntrinsicModelTest.cpp --------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::SmallVector;
using llvm::succeeded;

namespace {

struct TempDir {
  llvm::SmallString<256> path;

  ~TempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

static TargetProfile makeProfileWithIntrinsics() {
  TargetProfile profile;
  profile.identity.socVersion = "TestSoC";
  profile.hardware.aiCoreCount = 1;
  profile.hardware.l1SizeBytes = 512 * 1024;
  profile.hardware.ubSizeBytes = 192 * 1024;
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"f16f16f16", "s32s8s8"},
       {ExecutionUnit::Cube}});
  profile.intrinsics.push_back({"Intrinsic_vadd", {"f16", "f32"}});
  profile.intrinsics.push_back({"Intrinsic_vexp", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_vtranspose", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_vgather", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_vreduce", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_out2l1", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_l12l0a", {"f16"}});
  profile.intrinsics.push_back(
      {"Intrinsic_data_move_transpose_l12l0b", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_fix_pipe_l0c2out", {"f16"}});
  profile.intrinsics.push_back(
      {"Intrinsic_fix_pipe_unit_list", {"pre_conv", "pre_act"}});
  return profile;
}

static const TargetIntrinsicInfo *
findIntrinsic(const TargetProfile &profile, llvm::StringRef name) {
  for (const TargetIntrinsicInfo &intrinsic : profile.intrinsics)
    if (intrinsic.name == name)
      return &intrinsic;
  return nullptr;
}

static bool containsUnit(const TargetIntrinsicInfo &intrinsic,
                         ExecutionUnit unit) {
  return llvm::is_contained(intrinsic.units, unit);
}

static void writeSyntheticCannProfile(llvm::StringRef root) {
  llvm::SmallString<256> configDir(root);
  llvm::sys::path::append(configDir, "aarch64-linux", "data",
                          "platform_config");
  ASSERT_FALSE(llvm::sys::fs::create_directories(configDir));

  llvm::SmallString<256> iniPath(configDir);
  llvm::sys::path::append(iniPath, "SyntheticSoC.ini");

  std::error_code ec;
  llvm::raw_fd_ostream os(iniPath, ec);
  ASSERT_FALSE(ec) << ec.message();

  os << R"ini(
[version]
SoC_version=SyntheticSoC
Short_SoC_version=SYN
NpuArch=SYNTH

[SoCInfo]
ai_core_cnt=1
cube_core_cnt=1
vector_core_cnt=1
memory_size=1048576
support_bf16=true

[AICoreSpec]
l1_size=524288
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
ub_size=196608
support_fixpipe=true

[AICoreintrinsicDtypeMap]
0=Intrinsic_vbridge|float16
1=Intrinsic_data_move_l12l0a|f16

[CUBECoreintrinsicDtypeMap]
0=Intrinsic_vbridge|float32
1=Intrinsic_cube_only|s32s8s8

[VectorCoreintrinsicDtypeMap]
0=Intrinsic_vector_only|float16

[AICoreMemoryRates]
Intrinsic_data_move_out2l1=64
)ini";
}

} // namespace

TEST(AscendTargetIntrinsicModelTest, SupportsIntrinsicAndDtypes) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->supportsIntrinsic("Intrinsic_mmad"));
  EXPECT_FALSE(model->supportsIntrinsic("Intrinsic_missing"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "f16f16f16"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "s32s8s8"));
  EXPECT_FALSE(model->supportsDTypePattern("Intrinsic_mmad", "f16"));
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesExecutionUnits) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Cube),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Vector),
            (SmallVector<std::string>{"Intrinsic_vadd", "Intrinsic_vexp",
                                      "Intrinsic_vgather", "Intrinsic_vreduce",
                                      "Intrinsic_vtranspose"}));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::DMA),
            (SmallVector<std::string>{"Intrinsic_data_move_l12l0a",
                                      "Intrinsic_data_move_out2l1",
                                      "Intrinsic_data_move_transpose_l12l0b",
                                      "Intrinsic_fix_pipe_l0c2out"}));
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesMovementIntrinsicKinds) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::DirectCopy),
            (SmallVector<std::string>{"Intrinsic_data_move_l12l0a"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::Load2D),
            (SmallVector<std::string>{"Intrinsic_data_move_out2l1"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::Load2DTranspose),
            (SmallVector<std::string>{
                "Intrinsic_data_move_transpose_l12l0b"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::FixPipe),
            (SmallVector<std::string>{"Intrinsic_fix_pipe_l0c2out"}));
  EXPECT_TRUE(model->getIntrinsicsForPathKind(PathKind::QueueTransfer).empty());
}

TEST(AscendTargetIntrinsicModelTest, KeepsFixPipeMetadataOutOfMovementPaths) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->supportsIntrinsic("Intrinsic_fix_pipe_unit_list"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_fix_pipe_unit_list",
                                          "pre_conv"));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::DMA),
            (SmallVector<std::string>{"Intrinsic_data_move_l12l0a",
                                      "Intrinsic_data_move_out2l1",
                                      "Intrinsic_data_move_transpose_l12l0b",
                                      "Intrinsic_fix_pipe_l0c2out"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::FixPipe),
            (SmallVector<std::string>{"Intrinsic_fix_pipe_l0c2out"}));
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesComputeIntrinsicKinds) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::Matmul),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorAdd),
            (SmallVector<std::string>{"Intrinsic_vadd"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorExp),
            (SmallVector<std::string>{"Intrinsic_vexp"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorTranspose),
            (SmallVector<std::string>{"Intrinsic_vtranspose"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorGather),
            (SmallVector<std::string>{"Intrinsic_vgather"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorReduce),
            (SmallVector<std::string>{"Intrinsic_vreduce"}));
}

TEST(AscendTargetIntrinsicModelTest, UsesRecordedUnitsBeforeNameFallback) {
  TargetProfile profile;
  profile.intrinsics.push_back(
      {"Intrinsic_vadd", {"f16"}, {ExecutionUnit::Cube}});

  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(profile);
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Cube),
            (SmallVector<std::string>{"Intrinsic_vadd"}));
  EXPECT_TRUE(model->getIntrinsicsForUnit(ExecutionUnit::Vector).empty());
}

TEST(AscendTargetIntrinsicModelTest, MergesDuplicateCapabilities) {
  TargetProfile profile;
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"f16f16f16"}, {ExecutionUnit::Cube}});
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"s32s8s8"},
       {ExecutionUnit::Cube, ExecutionUnit::Vector}});

  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(profile);
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "f16f16f16"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "s32s8s8"));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Cube),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Vector),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
}

TEST(AscendTargetIntrinsicModelTest, LoaderMergesIntrinsicSectionsAndUnits) {
  TempDir temp;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("ascend-cann-profile",
                                                    temp.path));
  writeSyntheticCannProfile(temp.path);

  FailureOr<TargetProfile> profile =
      CannTargetProfileLoader::load(temp.path, "SyntheticSoC");
  ASSERT_TRUE(succeeded(profile));

  const TargetIntrinsicInfo *merged =
      findIntrinsic(*profile, "Intrinsic_vbridge");
  ASSERT_NE(merged, nullptr);
  EXPECT_EQ(merged->dtypes,
            (SmallVector<std::string>{"float16", "float32"}));
  EXPECT_TRUE(containsUnit(*merged, ExecutionUnit::Cube));
  EXPECT_TRUE(containsUnit(*merged, ExecutionUnit::Vector));

  const TargetIntrinsicInfo *cubeOnly =
      findIntrinsic(*profile, "Intrinsic_cube_only");
  ASSERT_NE(cubeOnly, nullptr);
  EXPECT_TRUE(containsUnit(*cubeOnly, ExecutionUnit::Cube));

  const TargetIntrinsicInfo *vectorOnly =
      findIntrinsic(*profile, "Intrinsic_vector_only");
  ASSERT_NE(vectorOnly, nullptr);
  EXPECT_TRUE(containsUnit(*vectorOnly, ExecutionUnit::Vector));

  const TargetIntrinsicInfo *rateOnly =
      findIntrinsic(*profile, "Intrinsic_data_move_out2l1");
  ASSERT_NE(rateOnly, nullptr);
  EXPECT_TRUE(containsUnit(*rateOnly, ExecutionUnit::DMA));
}
