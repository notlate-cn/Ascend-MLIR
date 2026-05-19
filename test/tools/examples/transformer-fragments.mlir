// REQUIRES: ascend_env
// RUN: bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment layernorm --log | FileCheck --check-prefix=LAYER %s
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment qkv --runtime-e2e --log | FileCheck --check-prefix=QKV %s
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment qkv_heads --runtime-e2e --log | FileCheck --check-prefix=QKVHEADS %s
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment qkv_project_heads --runtime-e2e --log | FileCheck --check-prefix=QKVPROJECTHEADS %s
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment attn_score --batch 2 --seq 16 --k 32 --runtime-e2e --log | FileCheck --check-prefix=ATTNSCORE %s

// LAYER: transformer_fragment.layernorm.full_codegen=pass
// LAYER: transformer_fragment.layernorm.phase5_translate=pass
// LAYER: transformer_fragment.layernorm.runtime_session=pass
// LAYER: transformer_fragment.layernorm.validation=pass

// QKV: transformer_fragment.qkv.full_codegen=pass
// QKV: transformer_fragment.qkv.phase5_translate=pass
// QKV: transformer_fragment.qkv.artifact_compile=pass
// QKV: transformer_fragment.qkv.runtime_session=pass
// QKV: transformer_fragment.qkv.validation=pass

// QKVHEADS: transformer_fragment.qkv_heads.full_codegen=pass
// QKVHEADS: transformer_fragment.qkv_heads.phase5_translate=pass
// QKVHEADS: transformer_fragment.qkv_heads.artifact_compile=pass
// QKVHEADS: transformer_fragment.qkv_heads.runtime_session=pass
// QKVHEADS: transformer_fragment.qkv_heads.validation=pass

// QKVPROJECTHEADS: transformer_fragment.qkv_project_heads.full_codegen=pass
// QKVPROJECTHEADS: transformer_fragment.qkv_project_heads.phase5_translate=pass
// QKVPROJECTHEADS: transformer_fragment.qkv_project_heads.artifact_compile=pass
// QKVPROJECTHEADS: transformer_fragment.qkv_project_heads.runtime_session=pass
// QKVPROJECTHEADS: transformer_fragment.qkv_project_heads.validation=pass

// ATTNSCORE: transformer_fragment.attn_score.full_codegen=pass
// ATTNSCORE: transformer_fragment.attn_score.phase5_translate=pass
// ATTNSCORE: transformer_fragment.attn_score.artifact_compile=pass
// ATTNSCORE: transformer_fragment.attn_score.runtime_session=pass
// ATTNSCORE: transformer_fragment.attn_score.validation=pass
