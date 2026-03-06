module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.consume}) {

    // Step 1: inline（作用在整个 module）
    transform.apply_registered_pass "inline" to %module
        : (!transform.any_op) -> !transform.any_op

    // Step 2: inline 之后重新 match，这时才拿 %func 的 handle
    %func = transform.structured.match
        ops{["func.func"]}
        attributes{sym_name = "fc_relu_spec"}
        in %module
        : (!transform.any_op) -> !transform.any_op

    // Step 3: 后续 pass 全部作用在 %func 上
    transform.apply_registered_pass "canonicalize" to %func
        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "cse" to %func
        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass
        "scf-for-loop-range-folding" to %func
        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "canonicalize" to %func
        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "cse" to %func
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}