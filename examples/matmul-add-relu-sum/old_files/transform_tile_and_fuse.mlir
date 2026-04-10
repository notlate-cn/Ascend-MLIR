module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // step1: 匹配出func.func
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // step2: 为func.func添加index args
    %transformed, %sz:3 = transform.func.add_index_args %0, 3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // step3: 匹配出linalg.matmul、linalg.elementwise、linalg.add和linalg.max
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %add, %max = transform.split_handle %elementwise : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // step4: 倒序进行TileAndFuse。先对linalg.max进行tile
    %tiled_max, %loop0, %loop1 = transform.structured.tile_using_for %max
      tile_sizes [%sz#0, %sz#1]
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // step5: 把add也fuse到loop1中(内层循环)
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // step6: 把linalg.matmul也fuse到loop1中(内层循环)，实际上这里返回的是3个matmul，#0是真正的matmul，#1和#2用于获取结果的维度，通过canonicalize可以消除
    %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op %matmul into %loop1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // step7: 针对matmul_fused_split#0进行k轴切分
    %matmul_fused_split:3 = transform.split_handle %matmul_fused : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %ret, %loops3 = transform.structured.tile_using_for %matmul_fused_split#0 tile_sizes [0, 0, %sz#2] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
