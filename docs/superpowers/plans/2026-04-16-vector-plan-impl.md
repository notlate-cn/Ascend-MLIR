# Vector Plan 实现方案（索引）

**Status**: 已拆分为每模块独立文档，见下方链接  
**Superseded by**: `docs/superpowers/plans/vector-plan/`

---

## 模块文档

| 文档 | 内容 | 前置 |
|------|------|------|
| [impl-00-foundation](vector-plan/impl-00-foundation.md) | 数据结构 header + Pass 骨架 + CMake | — |
| [impl-01-group-analysis](vector-plan/impl-01-group-analysis.md) | Pass 1：AxisLattice + CanFuse + 迭代融合 | impl-00 |
| [impl-02-group-outline](vector-plan/impl-02-group-outline.md) | Outline Pass：分桶 + outlining + 文件分离 | impl-01 |
| [impl-03-tile-fuse](vector-plan/impl-03-tile-fuse.md) | Pass 2：Collapse + TilePlan + LoopNest + Emit | impl-00 |
| [impl-04-tile-info](vector-plan/impl-04-tile-info.md) | TileInfo 接入层 + PrepareForEmit 适配 | impl-03 |

## 依赖图

```
impl-00 (foundation)
 ├─ impl-01 (group-analysis)
 │    └─ impl-02 (group-outline)
 └─ impl-03 (tile-fuse)        ← 可与 impl-01/02 并行开发
      └─ impl-04 (tile-info)
```

**建议开发顺序**：impl-00 → impl-03 → impl-01 → impl-02 → impl-04  
（impl-03 是核心路径，先落地验证 SliceComputer / GroupEmitter 的设计）

## 设计依据

`docs/superpowers/specs/2026-04-14-vector-plan-unified-design.md`
