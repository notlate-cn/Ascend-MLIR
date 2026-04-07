# Non-Mix Pipeline Regression Results

| Example | Expected Kind | Last Good Stage | First Failing Stage | Compile | `.bin` Exists | Runtime Initialized | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | yes | no | not run | environment | `/tmp/add-broadcast-concat.log`; compile succeeds and emits `.bin`, then executor init fails with `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file: No such file or directory`. |
| broadcast-add-reduce | vec | STAGE 4: Buffer Placement | STAGE 5: Linalg -> AscendC | no | no | no | not run | compile | `/tmp/broadcast-add-reduce.log`; `afir-opt --linalg-to-ascendc` aborts in `ComputeConversion.cpp:626` on `Operation::getResult` assertion, so no `.bin` is produced. |
| gather-elementwise-fusion | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | yes | no | not run | environment | `/tmp/gather-elementwise-fusion.log`; codegen and compile succeed, then executor init fails with the same missing `libascend_hal.so` dependency. |
| relu-broadcast-transpose | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | yes | no | not run | environment | `/tmp/relu-broadcast-transpose.log`; codegen and compile succeed, then executor init fails with the same missing `libascend_hal.so` dependency. |
| split-relu-brc-add-mul | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | yes | no | not run | environment | `/tmp/split-relu-brc-add-mul.log`; codegen and compile succeed, then executor init fails with the same missing `libascend_hal.so` dependency. |
