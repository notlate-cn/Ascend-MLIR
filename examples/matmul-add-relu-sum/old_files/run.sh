source env.sh

## 第一步：应用transform
#afir-opt fc_add_relu.mlir --transform-preload-library=transform-library-paths=transform_tile_and_fuse.mlir --transform-interpreter=entry-point=__transform_main | tee output_step1_tile_fuse.mlir
#
## 第二步：canonicalize
#afir-opt --canonicalize output_step1_tile_fuse.mlir | tee output_step2_canonicalize.mlir
#
## 第三步：转成3级调度
#afir-opt output_step2_canonicalize.mlir --transform-preload-library=transform-library-paths=claude_transform_fixed.mlir --transform-interpreter=entry-point=__transform_main --canonicalize --cse --scf-for-loop-canonicalization


# 一步转换
afir-opt fc_add_relu.mlir --transform-preload-library=transform-library-paths=transform_tile_and_fuse_3level-5.mlir --transform-interpreter=entry-point=__transform_main --canonicalize --cse --scf-for-loop-canonicalization