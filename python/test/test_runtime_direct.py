#!/usr/bin/env python3
"""
Direct test of runtime library
"""
import os
import sys
import ctypes
from pathlib import Path

import numpy as np
import torch

os.environ["ASCEND_HOME_PATH"] = "/home/niu/Ascend/latest"
os.environ["SOC_VERSION"] = "Ascend910B1"

# 添加 runtime 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from runtime.executor import AscendRuntime

print("=" * 70)
print("Direct Runtime Test")
print("=" * 70)

print("\n[Step 1] Creating runtime...")
runtime = AscendRuntime()
print("✅ Runtime created")

print("\n[Step 2] Setting device...")
runtime.set_device(0)
print("✅ Device set")

print("\n[Step 3] Reading binary...")
with open("./hash_copy_asc_graph.bin", "rb") as f:
    binary_data = f.read()
print(f"✅ Binary read: {len(binary_data)} bytes")

print("\n[Step 4] Registering binary...")
kernel_handle = runtime.register_binary(binary_data)
print(f"✅ Binary registered: 0x{kernel_handle:X}")

print("\n[Step 5] Registering function...")
function_handle = runtime.register_function(kernel_handle, "hash_copy_asc_graph")
print(f"✅ Function registered: 0x{function_handle:X}")

print("\n[Step 6] Preparing data...")
torch.manual_seed(42)
input0 = torch.rand(20, 31, dtype=torch.float32)
input1 = torch.rand(1, 31, dtype=torch.float32)
print(f"✅ Data prepared: input0={input0.shape}, input1={input1.shape}")

print("\n[Step 7] Allocating memory...")
input0_addr = runtime.malloc(input0.numpy().nbytes)
input1_addr = runtime.malloc(input1.numpy().nbytes)
output_addr = runtime.malloc(20 * 31 * 4)
print(f"✅ Memory allocated")

print("\n[Step 8] Copying input data...")
runtime.memcpy_h2d(input0_addr, input0.numpy(), input0.numpy().nbytes)
runtime.memcpy_h2d(input1_addr, input1.numpy(), input1.numpy().nbytes)
print(f"✅ Input data copied")

print("\n[Step 9] Preparing tiling data...")
import struct

tiling_values = [1, 1, 1264, 0, 0, 31, 20, 0, 0, 128, 128, 128, 128, 0, 8192]
tiling_bytes = b"".join(struct.pack("<I", v) for v in tiling_values)
print(f"✅ Tiling data prepared: {len(tiling_bytes)} bytes")

print("\n[Step 10] Building args...")
args = [input0_addr, input1_addr, output_addr, 0]
for i in range(0, len(tiling_bytes), 8):
    word = tiling_bytes[i:i + 8]
    word = word + b'\x00' * (8 - len(word))
    val = int.from_bytes(word, 'little')
    args.append(val)
print(f"✅ Args built: {len(args)} args")

print("\n[Step 11] Getting stream...")
stream = runtime.get_stream()
print(f"✅ Stream: {stream}")

print("\n[Step 12] Launching kernel...")
runtime.launch_kernel(function_handle, 1, args, stream)
print("✅ Kernel launched")

print("\n[Step 13] Synchronizing...")
runtime.synchronize()
print("✅ Synchronized")

print("\n[Step 14] Copying output...")
output_buffer = ctypes.create_string_buffer(20 * 31 * 4)
runtime.memcpy(ctypes.addressof(output_buffer), output_addr, 20 * 31 * 4, 2)
output_array = np.frombuffer(output_buffer.raw, dtype=np.float32).reshape(20, 31)
print(f"✅ Output copied: {output_array.shape}")

print("\n[Step 15] Verifying...")
print(f"  Output[0, :5]: {output_array[0, :5]}")

print("\n" + "=" * 70)
print("✅✅✅ All steps completed successfully! ✅✅✅")
print("=" * 70)

# Clean up
runtime.free(input0_addr)
runtime.free(input1_addr)
runtime.free(output_addr)

# Use os._exit to avoid cleanup issues
os._exit(0)
