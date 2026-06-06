# Legacy Frontend File Organization

This document used to describe the file organization for a prototype frontend
dialect.

That prototype is not the active boundary for Ascend V2 development. Current
compiler work should use the Ascend-owned layout:

- `include/Conversion/Ascend`
- `lib/Conversion/Ascend`
- `include/Target/Ascend`
- `include/Target/CannKernel`
- `lib/Target/Ascend`
- `lib/Target/CannKernel`
- `tools/ascend-mlir-opt`
- `tools/ascend-mlir-translate`
- `tools/ascend-debug`

Historical details remain available in git history. Do not use the old
frontend-dialect layout as the organizing model for new Ascend V2 work.
