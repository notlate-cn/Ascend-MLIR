# two-elewise-dyn-e2e

Dynamic-shape twin of `examples/two-elewise-e2e/`. Same two independent
elementwise **chains** (2 fused ops each), but every tensor is fully dynamic
3D `tensor<?x?x?xf16>` instead of static `4x4`.

## What it shows

`model.mlir` contains two chains in a single `func.func @model`:

  - `out0 = (a + b) * e`   → `kernel_group0` (`add,mul` fused)
  - `out1 = (c * d) + f`   → `kernel_group1` (`mul,add` fused)

Because all three axes are dynamic, `--afir-symbolize-shapes` turns the tiled
axis extent into a **symbolic** expression (`s0*s1*s2`) carried through to the
kernel, instead of the static case's constant `16`. The extents are resolved
from the input `.npy` shapes at launch (ShapeDerived tiling params, not
tunables), so autotune is skipped (`NETWORK_RUNNER_SKIP_AUTOTUNE=1`) and the
same kernel runs any shape.

Outputs use `tensor.empty(%d0,%d1,%d2)` (dynamic dims need the runtime sizes),
so there are no fixed init args — contrast the static case's `%i0`/`%i1`.

## Run

```bash
bash examples/two-elewise-dyn-e2e/run.sh
# pick a different concrete shape:
D0=4 D1=5 D2=16 bash examples/two-elewise-dyn-e2e/run.sh
```

Expected tail of output:

```
network.output[0]: max_diff=...  PASS
network.output[1]: max_diff=...  PASS
```
