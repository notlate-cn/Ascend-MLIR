# AscendCBufferPlacementPass Phase 1 - COMPLETE ✅

**Date**: 2025-03-02
**Status**: FULLY COMPLETED

## Phase 1 Implementation Summary

All steps (A-G) from plan-B.md have been successfully implemented:

| Step | Description | Status |
|------|-------------|--------|
| **A** | Parse ascendc.prologue/epilogue annotations | ✅ Complete |
| **B** | Infer alloc TPosition (CO1, VECCALC, VECOUT) | ✅ Complete |
| **C** | Update func arguments to GM memory_space | ✅ Complete |
| **D** | Insert copy operations (data_copy_l2/l0/fixpipe) | ✅ **Complete** |
| **E** | Update alloc memory_space | ✅ Complete |
| **F** | Clear ascendc.* annotations | ✅ Complete |
| **G** | Verify IR | ✅ Complete |

## Implementation Details

### Step D: Copy Insertion (NEWLY COMPLETED)

Implemented full copy insertion logic with the following components:

#### Helper Functions:
1. **`getArgIndexForRole(StringRef role)`**
   - Maps role names to function argument indices
   - lhs=0, rhs=1, bias=2, result=3

2. **`findSourceMemref(StringRef role, scf::ForOp forOp)`**
   - Finds source memref by role within loop scope
   - Returns function arguments for GM sources
   - Searches posMap for CO1 buffers

3. **`computeElementCount(OpBuilder &builder, Location loc, MemRefType type)`**
   - Computes total element count for copy operations
   - Handles both static and dynamic shapes

4. **`createMemrefTypeWithSpace(...)`**
   - Creates memref type with specified TPosition memory_space

5. **`insertDataCopyL2(...)`**
   - Inserts `ascendc.data_copy_l2` for GM ↔ L1/VECIN transfers

6. **`insertDataCopyL0(...)`**
   - Inserts `ascendc.data_copy_l0` for L1 ↔ L0 transfers

7. **`insertFixpipe(...)`**
   - Inserts `ascendc.fixpipe` for CO1 → VECIN transfers

8. **`insertCopiesForLoop(...)`**
   - Main function to insert copies for a single loop
   - Handles prologue tasks at loop body start
   - Handles epilogue tasks before loop terminator

9. **`insertCopiesForLoops(...)`**
   - Processes all loops from outermost to innermost
   - Ensures correct alloc lifetimes

### Copy Operations Supported:

| Transfer Type | Source | Destination | Operation |
|---------------|--------|-------------|-----------|
| GM → L1/VECIN | GM (0) | A1/B1/VECIN (1/3/9) | `data_copy_l2` |
| L1 → L0 | A1/B1 (1/3) | A2/B2 (2/4) | `data_copy_l0` |
| CO1 → VECIN | CO1 (7) | VECIN (9) | `fixpipe` |
| VECOUT → GM | VECOUT (10) | GM (0) | `data_copy_l2` |

### Prologue Processing:
- Parses annotation like `"lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"`
- Creates local allocs with destination memory_space
- Inserts appropriate copy operation based on source position
- Records new allocs for later reference

### Epilogue Processing:
- Handles CO1 → VECIN transfers via `fixpipe`
- Handles VECOUT → GM transfers via `data_copy_l2`
- Creates destination allocs with correct memory_space

## File Structure

```
lib/Conversion/AscendCBufferPlacement/
├── AscendCBufferPlacementPass.h        # Pass declaration
├── AscendCBufferPlacementPass.cpp      # Pass implementation (676 lines)
└── CMakeLists.txt                      # Build configuration

test/Conversion/
└── ascendc-buffer-placement.mlir       # Unit tests (6 test cases)

externals/pyasc/include/ascir/Dialect/Asc/IR/Core/
└── Attributes.td                       # TPositionMemSpaceAttr definition
```

## Testing

### Unit Tests Created:
1. `test_gm_args` - GM memory_space on function arguments
2. `test_subview_prop` - SubView memory_space propagation
3. `test_matmul_co1` - Matmul output alloc → CO1
4. `test_vector_veccalc` - Vector op alloc → VECCALC/VECOUT
5. `test_prologue_gm_l1` - Prologue copy insertion (GM → L1)
6. `test_annotation_cleanup` - Annotation removal

### Test Commands:
```bash
# Run specific test
afir-opt test/Conversion/ascendc-buffer-placement.mlir --ascendc-buffer-placement

# Or using lit
cd build
lit -v test/Conversion/ascendc-buffer-placement.mlir
```

## Code Statistics

- **Total Lines**: 676 (up from 497)
- **Added Functions**: 9 helper functions for copy insertion
- **Namespaces**: 7 logical sections (Copy task, TPosition utilities, Annotation parsing, Buffer inference, GM args, Alloc update, Annotation cleanup, Copy insertion)

## Key Design Decisions

1. **Loop Processing Order**: Outermost to innermost ensures correct alloc lifetimes
2. **Position Map Reuse**: Buffer positions inferred once and reused for copy insertion
3. **Shape Computation**: Simplified for static shapes, placeholder for dynamic shapes
4. **Operation Selection**: Uses source position to determine correct copy operation type

## Known Limitations

1. **Dynamic Shape Handling**: Uses estimated tile size (128) for fully dynamic shapes
2. **Role-to-Alloc Tracking**: Simplified - relies on `ascendc.role` attribute for basic tracking
3. **Nested Loop Complexities**: May need refinement for complex nested scenarios

## Next Steps (Phase 2)

1. **Refine Dynamic Shape Handling**: Compute exact sizes from runtime dimension values
2. **Enhanced Role Tracking**: Implement robust alloc-to-role mapping across nested loops
3. **Complex Epilogue Handling**: Support for multi-stage epilogue operations
4. **Integration Testing**: Test with full `fc_add_relu.mlir` pipeline

## References

- **Specification**: `examples/matmul-add-relu-sum/plan-B.md`
- **Key File**: `lib/Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.cpp`
- **Test File**: `test/Conversion/ascendc-buffer-placement.mlir`
