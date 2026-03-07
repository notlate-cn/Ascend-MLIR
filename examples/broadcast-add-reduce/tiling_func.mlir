// ============================================================
// tiling_func.mlir — Host-side tiling function for broadcast_add_reducesum
//
// Inputs:  %M  (rows, i64),  %N  (cols, i64)
// Outputs: TilingData struct with fields:
//            TB    — tile size along M for multi-core dispatch (level 1)
//            Tb    — tile size along M per AiCore (level 2)
//
// This function runs on the host CPU to compute tiling parameters.
// The inner algorithm is intentionally hardcoded (simplified from the
// heuristic sketch in broadcast-add.mlir) to demonstrate the pipeline
// structure.
//
// The reduction axis (N) is NOT tiled — each AiCore processes a contiguous
// slice of rows and reduces the full N columns in its inner loop.
// ============================================================

// TilingData layout (2 × i64 fields):
//   [0] TB  — outer tile (inter-core dispatch)
//   [1] Tb  — inner tile (per-core UB buffer)
!TilingData = !llvm.struct<"TilingData", (i64, i64)>

module {

  // ----------------------------------------------------------
  // @tiling_func: compute tiling parameters from runtime shape
  // ----------------------------------------------------------
  func.func @tiling_func(%M: i64, %N: i64) -> !TilingData {

    // ── Hardware constants ────────────────────────────────────
    %CORE_NUM  = arith.constant 20     : i64   // AiCore count
    %UB_BYTES  = arith.constant 262144 : i64   // 256 KB UB
    %ELEM_BYTES = arith.constant 2     : i64   // f16 = 2 bytes

    // ── Tb: rows that fit in UB for one row of B (Tb × N × f16) ─
    // We need 3 buffers (src A slice, B tile, result): UB / (3 × N × 2).
    // Clamp to at least 1.
    %c1   = arith.constant 1 : i64
    %c3   = arith.constant 3 : i64
    %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64   // 131072
    %per_buf     = arith.divsi %total_elems, %c3 : i64        // 43690
    // Rows = per_buf / N  (at least 1)
    %rows_raw    = arith.divsi %per_buf, %N : i64
    %rows_clamped = arith.maxsi %rows_raw, %c1 : i64
    %Tb = %rows_clamped : i64

    // ── TB: outer tile (one full AiCore workload) ─────────────
    // Set TB = Tb so each dispatch iteration maps 1:1 with AiCore work.
    %TB = %Tb : i64

    // ── Pack into TilingData struct ───────────────────────────
    %td0 = llvm.mlir.undef : !TilingData
    %td1 = llvm.insertvalue %TB, %td0[0] : !TilingData
    %td2 = llvm.insertvalue %Tb, %td1[1] : !TilingData

    return %td2 : !TilingData
  }

  // ----------------------------------------------------------
  // @runtime_dispatch: host-side launch entry point (conceptual)
  // ----------------------------------------------------------
  func.func @runtime_dispatch(
      %A:   memref<?xf16>,      // 1-D broadcast source (length M)
      %B:   memref<?x?xf16>,   // 2-D input matrix (M × N)
      %out: memref<?xf16>       // output reduction (length M)
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // ── Get runtime shapes ────────────────────────────────────
    %M_idx = memref.dim %A, %c0 : memref<?xf16>
    %N_idx = memref.dim %B, %c1 : memref<?x?xf16>
    %M = arith.index_cast %M_idx : index to i64
    %N = arith.index_cast %N_idx : index to i64

    // ── Compute tiling parameters ─────────────────────────────
    %tiling = func.call @tiling_func(%M, %N)
              : (i64, i64) -> !TilingData

    // ── Unpack ────────────────────────────────────────────────
    %TB = llvm.extractvalue %tiling[0] : !TilingData
    %Tb = llvm.extractvalue %tiling[1] : !TilingData

    // ── Launch kernel (conceptual) ────────────────────────────
    // acl_launch_kernel("broadcast_add_reducesum", core_num,
    //                   A, B, out, &tiling_data)

    return
  }

} // module
