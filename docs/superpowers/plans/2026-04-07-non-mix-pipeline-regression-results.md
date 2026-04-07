# Non-Mix Pipeline Regression Results

Legend: `yes` = observed success, `no` = observed failure, `not run` = the step was not reached.

| Example | Expected Kind (Plan) | Last Good Stage | First Failing Stage | Compile | `.bin` Exists | Runtime Initialized | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/add-broadcast-concat.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| broadcast-add-reduce | vec | STAGE 4: Buffer Placement | STAGE 5: `--linalg-to-ascendc` reduction-generic lowering | no | no | no | not run | compile | xvm@orb / 2026-04-07; `/tmp/broadcast-add-reduce.log`; sig: `afir-opt --linalg-to-ascendc` aborts at `ComputeConversion.cpp:626`; reduction-generic failure, so no `.bin`. |
| gather-elementwise-fusion | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/gather-elementwise-fusion.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| relu-broadcast-transpose | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/relu-broadcast-transpose.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
| split-relu-brc-add-mul | vec | STAGE 9: Compile | STAGE 10: executor init before kernel launch | yes | yes | no | not run | environment | xvm@orb / 2026-04-07; `/tmp/split-relu-brc-add-mul.log`; sig: `dlopen failed (.../libascendcl.so): libascend_hal.so: cannot open shared object file`. |
