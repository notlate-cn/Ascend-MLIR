// -----// IR Dump After VectorPlanGroupOutline (vector-plan-group-outline) //----- //
module {
  func.func private @kernel_group0(tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.func private @kernel_group1(tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.func @bucketed_dyn(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>, %arg4: tensor<?x?x?xf32>, %arg5: tensor<?x?x?xf32>, %arg6: tensor<?x?x?xf32>, %arg7: tensor<?x?xf32>, %arg8: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %0 = call @kernel_group0(%arg0, %arg1, %arg2, %arg3, %arg7) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %1 = call @kernel_group1(%arg4, %arg5, %arg6, %arg8) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
}


