# Non-Mix Pipeline Regression Results

Legend: `yes` = observed success, `no` = observed failure, `not run` = the step was not reached.

Task 2 routing inspection on xvm (`2026-04-07`): all five `step7_cann.mlir` artifacts are still vector-classified on the inspected compute ops (`ascendc.unit = "AiCore.Vector"`), with no `ascendc.kernel_kind = "mix"` evidence in the checked files. The corresponding `step8_kernel.cpp` artifacts also contain no `KERNEL_TYPE_MIX`, `ASCEND_IS_AIC`, `ASCEND_IS_AIV`, `CrossCoreSetFlag`, or `CrossCoreWaitFlag` markers. This supports keeping the non-mix routing boundary unchanged.
`lib/Target/CannKernel/CannTranslation.cpp` was left unchanged because this inspection did not prove any non-mix routing leakage into mix-specific lowering.

Task 2 verification follow-up on xvm:
- `cmake --build build --target afir-translate -j4` was already up to date (`ninja: no work to do`).
- `source examples/env.sh && bash examples/relu-broadcast-transpose/run.sh --log` reached Stage 10 and failed only at executor initialization with the known `libascend_hal.so` missing-library error.
- `source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log` completed successfully, and the mix validator still passed with `PASS`.

Task 3 runtime separation inspection:
- `lib/Runtime/Executor.cpp` now loads only the RT simulator library during `Initialize()`. ACL libraries and `RunPackedMixFile()`-only symbols are loaded lazily inside the packed-mix path, so plain `RegisterBinary` / `RunWithHandle` execution no longer requires `libascendcl.so` up front.
- `tools/validator/validator_main.cpp` already gates packed execution on `--kernel-type mix`; the plain validator path still uses `RegisterBinary` and therefore stays on the non-mix execution route.
- `lib/Runtime/Compiler.cpp` still distinguishes mix compilation only when `kernel_type == "mix"`, so the five non-mix examples remain on the plain `.bin` generation path.
- Controller-side xvm rerun after the `Executor.cpp` boundary fix confirmed the effect: the four former environment-blocked examples now initialize the runtime, execute the plain path, and pass validation under their existing tolerances.

Task 4 shared environment-contract assessment:
- The earlier `libascend_hal.so` failure was an environment-only symptom at executor initialization, not a codegen regression. After the Task 3 runtime-boundary fix, the previously blocked examples now pass on xvm, so there is no remaining shared environment-script regression to repair in `examples/env.sh`.
- Follow-up xvm rerun after the `ComputeConversion.cpp` reduction fix confirmed the remaining gap is also closed: `broadcast-add-reduce` now completes `--linalg-to-ascendc`, emits a binary, and passes its existing end-to-end validation tolerance.

| Example | Expected Kind (Plan) | Last Good Stage | First Failing Stage | Compile Status | Runtime Status | Accuracy Status | Final Root Cause | Issue State | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | PASS | none | pass | pass | pass | none | fixed | xvm@orb / 2026-04-07 rerun; validation passed with `max_abs_diff=0`, `mean_abs_diff=0`. |
| broadcast-add-reduce | vec | PASS | none | pass | pass | pass | none | fixed | xvm@orb / 2026-04-07 rerun after `ComputeConversion.cpp` fix; validation passed with `max_abs_diff=4.800000e+01`, `mean_abs_diff=8.069922e+00`. |
| gather-elementwise-fusion | vec | PASS | none | pass | pass | pass | none | fixed | xvm@orb / 2026-04-07 rerun; validation passed with `max_abs_diff=1.494141e+00`, `mean_abs_diff=4.083161e-01`. |
| relu-broadcast-transpose | vec | PASS | none | pass | pass | pass | none | fixed | xvm@orb / 2026-04-07 rerun; validation passed with `max_abs_diff=0`, `mean_abs_diff=0`. |
| split-relu-brc-add-mul | vec | PASS | none | pass | pass | pass | none | fixed | xvm@orb / 2026-04-07 rerun; validation passed with `max_abs_diff=9.765625e-04`, `mean_abs_diff=2.629566e-05`. |

Task 5 final verification:
- `bash examples/matmul-add-leakyrelu/run.sh --log` on xvm completed successfully with `max_abs_diff=0`, `mean_abs_diff=0`, and `PASS`.
- The five non-mix examples were rerun under `source examples/env.sh`; all five now pass end-to-end on xvm under their current validation thresholds.

## Recovery Summary

- fixed shared translator routing: no
- fixed shared runtime routing: yes
- fixed shared environment contract: no
- sample-specific patches introduced: no
- mix regression check preserved: yes
