source env.sh

# 第一步：转换命令，命令结果同时打屏和落盘到 output_step1_tile_and_fuse_3level.mlir 文件
afir-opt fc_add_relu.mlir --transform-preload-library=transform-library-paths=transform_tile_and_fuse_3level.mlir --transform-interpreter=entry-point=__transform_main --canonicalize --cse | tee output_step1_tile_and_fuse_3level.mlir

# 第二步：Bufferize
afir-opt output_step1_tile_and_fuse_3level.mlir --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops=true" --buffer-deallocation-pipeline  | tee output_step2_bufferized.mlir

# 第三步：Buffer Placement，标记分配位置，同时进行内存分配
afir-opt output_step2_bufferized.mlir --ascendc-buffer-placement --canonicalize --cse | tee output_step3_buffer_placement.mlir

# 第四步：Lowering到PyAsc
afir-opt output_step3_buffer_placement.mlir --linalg-to-ascendc --canonicalize --cse --debug | tee output_step4_lowering_to_asc.mlir

#
#afir-opt output_step4_lowering_to_asc.mlir \
#    --canonicalize \
#    --lower-affine \
#    --canonicalize \
#    -o output_stepE1_shape_specialized.mlir
#
#afir-opt output_stepE1_shape_specialized.mlir \
#    --scf-for-loop-range-folding \
#    --canonicalize \
#    -o output_stepE2_optimized.mlir
#
#
#afir-opt merged_for_spec.mlir \
#    --inline="default-pipeline='' max-iterations=1 inlining-threshold=99999999" \
#    --canonicalize \
#    --cse \
#    --scf-for-loop-range-folding \
#    --canonicalize \
#    | tee output_stepE3_specialized.mlir


#afir-opt merged_for_spec.mlir \
#    --canonicalize \
#    --scf-for-loop-peeling \
#    --scf-for-loop-range-folding \
#    --canonicalize \
#    --cse \
#    | tee output_stepE3_specialized.mlir