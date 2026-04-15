#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS="${RUNS:-3}"
PYPTO_ROOT="${PYPTO_ROOT:-/home/niu/code/pypto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --pypto-root)
      PYPTO_ROOT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${PYPTO_ROOT}/python/pypto/pypto_impl.cpython-310-aarch64-linux-gnu.so" ]]; then
  echo "PyPTO python extension not found under ${PYPTO_ROOT}/python" >&2
  exit 1
fi

echo "ascend_mlir_mix_compile"
RUNS="${RUNS}" bash "${SCRIPT_DIR}/run_mix_compile_timing_compare.sh"

PY_BENCH="$(mktemp /tmp/pypto_mix_timing.XXXXXX.py)"
trap 'rm -f "${PY_BENCH}"' EXIT

cat >"${PY_BENCH}" <<'PY'
import argparse
import statistics
import time

import pypto
from pypto import pypto_impl
import torch


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM}, use_cache=False)
def matmul_add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [a.shape[0], a.shape[1]], [0, 0])
        b0 = pypto.view(b, [b.shape[0], b.shape[1]], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def make_tensors(m, k, n):
    return (
        torch.ones((m, k), dtype=torch.int8),
        torch.ones((k, n), dtype=torch.int8),
        torch.ones((m, n), dtype=torch.int32),
        torch.zeros((m, n), dtype=torch.int32),
    )


def compile_only(m, k, n):
    tensors = make_tensors(m, k, n)
    in_tensors, non_tensor_values, input_tensor_defs = matmul_add_kernel._parse_call_args(tensors, {})
    start = time.perf_counter()
    matmul_add_kernel._get_or_create_kmodule(non_tensor_values)
    pto_tensors = matmul_add_kernel._convert_tensors_with_metadata(in_tensors, input_tensor_defs)
    with pypto.options("jit_scope"):
        matmul_add_kernel._set_config_option()
        pypto_impl.DeviceInit()
        matmul_add_kernel.compile(pto_tensors)
    return time.perf_counter() - start


def compile_and_sim(m, k, n):
    start = time.perf_counter()
    matmul_add_kernel(*make_tensors(m, k, n))
    return time.perf_counter() - start


def run(label, fn, args):
    times = []
    for i in range(args.runs):
        elapsed = fn(args.m, args.k, args.n)
        times.append(elapsed)
        print(f"run mode={label} index={i + 1} wall_ms={elapsed * 1000:.3f}", flush=True)
    print(
        "mode="
        f"{label} case=matmul_add_int8_i32_sim shape={args.m}x{args.k}x{args.n} "
        f"runs={len(times)} mean_ms={statistics.mean(times) * 1000:.3f} "
        f"min_ms={min(times) * 1000:.3f} max_ms={max(times) * 1000:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()
    run("pypto_compile_only_private", compile_only, args)
    run("pypto_compile_and_sim_public", compile_and_sim, args)


if __name__ == "__main__":
    main()
PY

cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env >/dev/null

echo "pypto_mix_timing"
cd "${PYPTO_ROOT}"
PYTHONPATH="${PYPTO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${PY_BENCH}" --runs "${RUNS}"
