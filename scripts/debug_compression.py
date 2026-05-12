#!/usr/bin/env python3
"""
Debug KV Cache Compression

Test if baseline compression actually reduces cache size.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
from baselines.unified_interface import (
    BaselineMethod,
    CacheConfig,
    create_baseline,
)

print("=" * 60)
print("Debug: KV Cache Compression")
print("=" * 60)

# Create synthetic KV cache
batch = 1
num_heads = 8
seq_len = 500
head_dim = 128

keys = torch.randn(batch, num_heads, seq_len, head_dim)
values = torch.randn(batch, num_heads, seq_len, head_dim)

print(f"Original cache size: {keys.shape}")

# Test each baseline
configs = {
    'H2O': CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1),
    'StreamingLLM': CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=50),
    'KIVI': CacheConfig(method=BaselineMethod.KIVI, quantize_bits=4),
    'Random': CacheConfig(method=BaselineMethod.RANDOM, heavy_ratio=0.1, recent_ratio=0.1),
}

for name, config in configs.items():
    print(f"\nTesting {name}...")

    baseline = create_baseline(config)
    baseline.reset()

    # Simulate attention weights
    attn_weights = torch.randn(batch, num_heads, 1, seq_len).softmax(dim=-1)

    # First compression (prefill)
    ck, cv = baseline.compress(keys, values, attn_weights, seq_len - 1)

    print(f"  After compress: {ck.shape}")
    print(f"  Compression ratio: {ck.shape[2] / seq_len:.2%}")
    print(f"  Stats: {baseline.get_stats()}")

    # Multiple compressions (simulate decode)
    for step in range(5):
        # Add new tokens
        new_len = ck.shape[2] + 10
        new_keys = torch.randn(batch, num_heads, 10, head_dim)
        new_values = torch.randn(batch, num_heads, 10, head_dim)

        keys = torch.cat([ck, new_keys], dim=2)
        values = torch.cat([cv, new_values], dim=2)

        # New attention
        new_attn = torch.randn(batch, num_heads, 1, new_len).softmax(dim=-1)

        # Compress
        ck, cv = baseline.compress(keys, values, new_attn, new_len - 1)

    print(f"  After 5 iterations: {ck.shape}")
    print(f"  Final ratio: {ck.shape[2] / (seq_len + 50):.2%}")

print("\n" + "=" * 60)
print("Debug Complete!")
print("=" * 60)