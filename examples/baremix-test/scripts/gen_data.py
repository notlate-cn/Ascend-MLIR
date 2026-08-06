#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np


def gen_golden_data():
    m = 128
    n = 128
    k = 256

    input_a = np.random.randint(-10, 10, [m, k]).astype(np.float16)
    input_b = np.random.randint(-10, 10, [k, n]).astype(np.float16)
    input_bias = np.random.randint(1, 10, [n]).astype(np.float32)
    alpha = 0.001
    golden = (
        np.matmul(input_a.astype(np.float32), input_b.astype(np.float32))
        + input_bias
    ).astype(np.float32)
    golden = np.where(golden >= 0, golden, golden * alpha)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
