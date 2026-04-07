# Non-Mix Pipeline Regression Results

Legend: `yes` = observed success, `no` = observed failure, `not run` = the step was not reached.

Task 2 routing inspection on xvm (`2026-04-07`): all five `step7_cann.mlir` artifacts are still vector-classified on the inspected compute ops (`ascendc.unit = "AiCore.Vector"`), with no `ascendc.kernel_kind = "mix"` evidence in the checked files. The corresponding `step8_kernel.cpp` artifacts also contain no `KERNEL_TYPE_MIX`, `ASCEND_IS_AIC`, `ASCEND_IS_AIV`, `CrossCoreSetFlag`, or `CrossCoreWaitFlag` markers. This supports keeping the non-mix routing boundary unchanged; the only confirmed functional failure remains `broadcast-add-reduce` at `STAGE 5 --linalg-to-ascendc`.

| Example | Expected Kind (Plan) | Last Good Stage | First Failing Stage | Compile | `.bin` Exists | Runtime Initialized | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/add-broadcast-concat.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| broadcast-add-reduce | vec | STAGE 4: Buffer Placement | STAGE 5: `--linalg-to-ascendc` reduction-generic lowering | no | no | no | not run | compile | xvm@orb / 2026-04-07; `/tmp/broadcast-add-reduce.log`; sig: `afir-opt --linalg-to-ascendc` aborts at `ComputeConversion.cpp:626`; reduction-generic failure, so no `.bin`. |
| gather-elementwise-fusion | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/gather-elementwise-fusion.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| relu-broadcast-transpose | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/relu-broadcast-transpose.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| split-relu-brc-add-mul | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/split-relu-brc-add-mul.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
