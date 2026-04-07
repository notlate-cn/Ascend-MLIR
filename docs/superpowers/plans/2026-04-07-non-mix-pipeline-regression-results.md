# Non-Mix Pipeline Regression Results

| Example | Expected Kind | Last Good Stage | First Failing Stage | Compile | Runtime | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | no | no | environment | Compile succeeds and produces `.bin`; executor init fails with `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file: No such file or directory`. |
| broadcast-add-reduce | vec | STAGE 4: Buffer Placement | STAGE 5: Linalg -> AscendC | no | no | no | compile | `afir-opt --linalg-to-ascendc` aborts in `ComputeConversion.cpp:626` on `Operation::getResult` assertion; no `.bin` is produced. |
| gather-elementwise-fusion | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | no | no | environment | Codegen and compile succeed; executor init fails with the same missing `libascend_hal.so` dependency. |
| relu-broadcast-transpose | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | no | no | environment | Codegen and compile succeed; executor init fails with the same missing `libascend_hal.so` dependency. |
| split-relu-brc-add-mul | vec | STAGE 9: Compile | STAGE 10: Run + Verify | yes | no | no | environment | Codegen and compile succeed; executor init fails with the same missing `libascend_hal.so` dependency. |
