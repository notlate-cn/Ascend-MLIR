#!/usr/bin/env python3
import argparse
import collections
import statistics
import time

import pypto
from pypto.frontend.parser.entry import JitCallableWrapper
import torch


stage_times = collections.defaultdict(list)


def wrap_method(name):
    original = getattr(JitCallableWrapper, name)

    def wrapped(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            stage_times[name].append((time.perf_counter() - start) * 1000.0)

    setattr(JitCallableWrapper, name, wrapped)


def wrap_static_method(name):
    original = getattr(JitCallableWrapper, name)

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stage_times[name].append((time.perf_counter() - start) * 1000.0)

    setattr(JitCallableWrapper, name, staticmethod(wrapped))


for method_name in [
    "_parse_call_args",
    "_get_or_create_kmodule",
    "_set_config_option",
    "compile",
    "_run_with_cpu",
    "_execute_kernel",
]:
    wrap_method(method_name)

wrap_static_method("_convert_tensors_with_metadata")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()

    totals = []
    previous_counts = {}
    for index in range(args.runs):
        start = time.perf_counter()
        matmul_add_kernel(*make_tensors(args.m, args.k, args.n))
        total = (time.perf_counter() - start) * 1000.0
        totals.append(total)

        print(f"run={index + 1} public_total_ms={total:.3f}")
        for name in sorted(stage_times):
            values = stage_times[name][previous_counts.get(name, 0):]
            previous_counts[name] = len(stage_times[name])
            if not values:
                continue
            joined = ",".join(f"{value:.3f}" for value in values)
            print(
                f"  stage={name} calls={len(values)} "
                f"sum_ms={sum(values):.3f} values_ms={joined}"
            )

    print("summary")
    print(
        f"stage=public_total mean_ms={statistics.mean(totals):.3f} "
        f"min_ms={min(totals):.3f} max_ms={max(totals):.3f}"
    )
    for name in sorted(stage_times):
        values = stage_times[name]
        print(
            f"stage={name} calls={len(values)} "
            f"mean_ms={statistics.mean(values):.3f} "
            f"sum_mean_per_run_ms={sum(values) / args.runs:.3f}"
        )


if __name__ == "__main__":
    main()
