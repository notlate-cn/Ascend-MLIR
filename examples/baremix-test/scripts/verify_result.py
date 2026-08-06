#!/usr/bin/python3
# coding=utf-8

import sys

import numpy as np

relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4


def verify_result(output, golden):
    output_data = np.fromfile(output, dtype=np.float32).reshape(-1)
    golden_data = np.fromfile(golden, dtype=np.float32).reshape(-1)
    compare = np.isclose(
        output_data,
        golden_data,
        rtol=relative_tol,
        atol=absolute_tol,
        equal_nan=True,
    )
    diff_indices = np.where(compare == False)[0]
    for index, real_index in enumerate(diff_indices):
        expected = golden_data[real_index]
        actual = output_data[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
            % (real_index, expected, actual, abs(actual - expected) / expected)
        )
        if index == 100:
            break

    error_ratio = float(diff_indices.size) / golden_data.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        passed = verify_result(sys.argv[1], sys.argv[2])
        if not passed:
            raise ValueError("[ERROR] result error")
        print("test pass")
    except Exception as exc:
        print(exc)
        sys.exit(1)
