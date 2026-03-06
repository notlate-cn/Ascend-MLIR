#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  func.func @fc_relu_tb(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64) -> tensor<?x?xf32> {
    // arg0: 左矩阵 (M×K)
    // arg1: 右矩阵 (K×N)
    // arg2: 偏置矩阵 (M×N)
    // arg3: 输出矩阵 (M×N)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32

    %step_m = arith.index_cast %arg4 : i64 to index
    %step_n = arith.index_cast %arg5 : i64 to index
    %step_k = arith.index_cast %arg6 : i64 to index

    // 构造全0矩阵，用于ReLU
    %dim_m = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_n = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %zeros = linalg.fill ins(%cst : f32) outs(tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>) -> tensor<?x?xf32>

    // ==== 外层TB循环 (M大Tile) ====
    %tb_m = scf.for %i_tb = %c0 to %dim_m step %step_m iter_args(%acc = %arg3) -> (tensor<?x?xf32>) {
      // TB N循环
      %tb_n = scf.for %j_tb = %c0 to %dim_n step %step_n iter_args(%acc1 = %acc) -> (tensor<?x?xf32>) {

        // 计算本Tile大小
        %cur_m = affine.min #map(%i_tb)[%dim_m, %step_m]
        %cur_n = affine.min #map(%j_tb)[%dim_n, %step_n]

        // ==== 中层Tb循环 (M小Tile) ====
        %tb_m_inner = scf.for %i_tb_inner = %c0 to %cur_m step %step_m iter_args(%acc2 = %acc1) -> (tensor<?x?xf32>) {
          %tb_n_inner = scf.for %j_tb_inner = %c0 to %cur_n step %step_n iter_args(%acc3 = %acc2) -> (tensor<?x?xf32>) {

            // 计算当前小Tile大小
            %tile_m = affine.min #map(%i_tb_inner)[%cur_m, %step_m]
            %tile_n = affine.min #map(%j_tb_inner)[%cur_n, %step_n]

            // ==== 内层t循环 (向量级/线程) ====
            %t_m = scf.for %i_t = %c0 to %tile_m step 1 iter_args(%acc4 = %acc3) -> (tensor<?x?xf32>) {
              %t_n = scf.for %j_t = %c0 to %tile_n step 1 iter_args(%acc5 = %acc4) -> (tensor<?x?xf32>) {

                // K轴切分逻辑保持不变
                %dim_k = tensor.dim %arg0, %c1 : tensor<?x?xf32>
                %mat_left = tensor.extract_slice %arg0[%i_tb_inner, 0] [%tile_m, %dim_k] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %mat_right = tensor.extract_slice %arg1[0, %j_tb_inner] [%dim_k, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %out_tile = tensor.extract_slice %acc5[%i_tb_inner, %j_tb_inner] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

                // K轴切分循环
                %mat_res = scf.for %k = %c0 to %dim_k step %step_k iter_args(%acc_k = %out_tile) -> (tensor<?x?xf32>) {
                  %cur_k = affine.min #map(%k)[%dim_k, %step_k]
                  %slice_l = tensor.extract_slice %mat_left[0, %k] [%tile_m, %cur_k] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                  %slice_r = tensor.extract_slice %mat_right[%k, 0] [%cur_k, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                  %slice_acc = tensor.extract_slice %acc_k[0, 0] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                  %matmul_res = linalg.matmul ins(%slice_l, %slice_r : tensor<?x?xf32>, tensor<?x?xf32>) outs(%slice_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %acc_insert = tensor.insert_slice %matmul_res into %acc_k[0, 0] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
                  scf.yield %acc_insert : tensor<?x?xf32>
                }

                // 加偏置
                %bias_tile = tensor.extract_slice %arg2[%i_tb_inner, %j_tb_inner] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %add_res = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%mat_res, %bias_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>

                // ReLU
                %zero_tile = tensor.extract_slice %zeros[%i_tb_inner, %j_tb_inner] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %relu_res = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> ins(%add_res, %zero_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
                %acc_insert_final = tensor.insert_slice %relu_res into %acc5[%i_tb_inner, %j_tb_inner] [%tile_m, %tile_n] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

                scf.yield %acc_insert_final : tensor<?x?xf32>
              }
              scf.yield %t_n : tensor<?x?xf32>
            }

            scf.yield %t_m : tensor<?x?xf32>
          }
          scf.yield %tb_n_inner : tensor<?x?xf32>
        }

        scf.yield %tb_m_inner : tensor<?x?xf32>
      }
      scf.yield %tb_n : tensor<?x?xf32>
    }
    return %tb_m : tensor<?x?xf32>
  }
}