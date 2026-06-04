#!/usr/bin/env bash

REAL_NPU_MICROCASES=(
  const640
  copy640
  copy_tbuf640
  copy_scalar640
  copy_params640
  copy_wait640
  const_with_input640
  relu_only
  broadcast_add
)

REAL_NPU_RELU_DIAGNOSTIC_MICROCASES=(
  relu_diag_broadcast_store
  relu_diag_strided_copy
  relu_diag_broadcast_add
  relu_diag_generated_buffers
  relu_diag_tiling_abi
)
