# Non-Mix Pipeline Regression Results

Legend: `yes` = observed success, `no` = observed failure, `not run` = the step was not reached.

Task 2 routing inspection on xvm (`2026-04-07`): all five `step7_cann.mlir` artifacts are still vector-classified on the inspected compute ops (`ascendc.unit = "AiCore.Vector"`), with no `ascendc.kernel_kind = "mix"` evidence in the checked files. The corresponding `step8_kernel.cpp` artifacts also contain no `KERNEL_TYPE_MIX`, `ASCEND_IS_AIC`, `ASCEND_IS_AIV`, `CrossCoreSetFlag`, or `CrossCoreWaitFlag` markers. This supports keeping the non-mix routing boundary unchanged; the only confirmed functional failure remains `broadcast-add-reduce` at `STAGE 5 --linalg-to-ascendc`.
`lib/Target/CannKernel/CannTranslation.cpp` was left unchanged because this inspection did not prove any non-mix routing leakage into mix-specific lowering.

Task 2 verification follow-up on xvm:
- `cmake --build build --target afir-translate -j4` was already up to date (`ninja: no work to do`).
- `source examples/env.sh && bash examples/relu-broadcast-transpose/run.sh --log` reached Stage 10 and failed only at executor initialization with the known `libascend_hal.so` missing-library error.
- `source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log` completed successfully, and the mix validator still passed with `PASS`.

Task 3 runtime separation inspection:
- `lib/Runtime/Executor.cpp` now loads only the RT simulator library during `Initialize()`. ACL libraries and `RunPackedMixFile()`-only symbols are loaded lazily inside the packed-mix path, so plain `RegisterBinary` / `RunWithHandle` execution no longer requires `libascendcl.so` up front.
- `tools/validator/validator_main.cpp` already gates packed execution on `--kernel-type mix`; the plain validator path still uses `RegisterBinary` and therefore stays on the non-mix execution route.
- `lib/Runtime/Compiler.cpp` still distinguishes mix compilation only when `kernel_type == "mix"`, so the five non-mix examples remain on the plain `.bin` generation path.
- Local rebuild verification was blocked by an incomplete LLVM build tree in this workspace: the generated Ninja graph depends on `libLLVMSupport.a`, but only partial LLVM shared objects are present here, so `cmake --build build --target compiler validator -j4` could not complete locally. The runtime boundary fix is still justified by code inspection, but the requested xvm rerun must be done in an environment with the full LLVM runtime artifacts.

| Example | Expected Kind (Plan) | Last Good Stage | First Failing Stage | Compile | `.bin` Exists | Runtime Initialized | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/add-broadcast-concat.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| broadcast-add-reduce | vec | STAGE 4: Buffer Placement | STAGE 5: `--linalg-to-ascendc` reduction-generic lowering | no | no | no | not run | compile | xvm@orb / 2026-04-07; `/tmp/broadcast-add-reduce.log`; sig: `afir-opt --linalg-to-ascendc` aborts at `ComputeConversion.cpp:626`; reduction-generic failure, so no `.bin`. |
| gather-elementwise-fusion | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/gather-elementwise-fusion.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| relu-broadcast-transpose | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/relu-broadcast-transpose.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| split-relu-brc-add-mul | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/split-relu-brc-add-mul.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
