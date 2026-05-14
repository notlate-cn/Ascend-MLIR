#!/usr/bin/env bash
set -euo pipefail

passes_td="${1:?usage: check_ascend_dependent_dialects.sh <Passes.td>}"

check_pass() {
  local pass_name="$1"
  shift

  local block
  block="$(awk "/def ${pass_name} /,/^}/" "$passes_td")"
  if [[ -z "$block" ]]; then
    echo "missing pass definition: ${pass_name}" >&2
    exit 1
  fi

  if ! grep -q "let dependentDialects" <<<"$block"; then
    echo "${pass_name}: missing dependentDialects" >&2
    exit 1
  fi

  local dialect
  for dialect in "$@"; do
    if ! grep -q "\"${dialect}\"" <<<"$block"; then
      echo "${pass_name}: missing dependent dialect ${dialect}" >&2
      exit 1
    fi
  done
}

check_pass AscendNormalizePass \
  "mlir::func::FuncDialect" \
  "mlir::tensor::TensorDialect" \
  "mlir::linalg::LinalgDialect" \
  "mlir::arith::ArithDialect" \
  "mlir::math::MathDialect" \
  "mlir::cf::ControlFlowDialect"

check_pass AscendKernelizePass \
  "mlir::func::FuncDialect" \
  "mlir::tensor::TensorDialect" \
  "mlir::linalg::LinalgDialect" \
  "mlir::arith::ArithDialect" \
  "mlir::cf::ControlFlowDialect"

check_pass AscendSchedulePass \
  "mlir::func::FuncDialect" \
  "mlir::tensor::TensorDialect" \
  "mlir::linalg::LinalgDialect" \
  "mlir::arith::ArithDialect"

check_pass AscendRealizePass \
  "mlir::bufferization::BufferizationDialect" \
  "mlir::memref::MemRefDialect" \
  "mlir::func::FuncDialect" \
  "mlir::tensor::TensorDialect" \
  "mlir::linalg::LinalgDialect" \
  "mlir::arith::ArithDialect"
