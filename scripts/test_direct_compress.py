#!/usr/bin/env python3
"""
Direct Compression Test

Test baseline compress directly on real model KV cache.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baselines.unified_interface import (
    BaselineMethod,
    CacheConfig,
    create_baseline,
)

print("=" * 60)
print("Direct Compression Test")
print("=" * 60)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# Create long prompt
text = """The development of artificial intelligence has been one of the most significant technological achievements.
Machine learning algorithms have revolutionized how we process data.
Deep learning has enabled breakthroughs in image recognition.
Natural language processing has transformed communication.
""" * 50

prompt = text + "\n\nSummary: "

# Encode
inputs = tokenizer(prompt, return_tensors='pt')
input_ids = inputs['input_ids'].to(model.device)
print(f"Input length: {input_ids.shape[1]} tokens")

# Run prefill
print("\nRunning prefill...")
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

# Check cache structure
print(f"Past key values type: {type(past_key_values)}")
print(f"Has key_cache: {hasattr(past_key_values, 'key_cache')}")
if hasattr(past_key_values, 'key_cache'):
    print(f"Number of layers: {len(past_key_values.key_cache)}")
    print(f"Key cache[0] shape: {past_key_values.key_cache[0].shape}")
    cache_len = past_key_values.key_cache[0].shape[2]
else:
    print("DynamicCache has no key_cache attribute!")
    cache_len = 0

# Test compression
print("\n" + "=" * 60)
print("Testing Compression")
print("=" * 60)

baselines = {
    'StreamingLLM': CacheConfig(
        method=BaselineMethod.STREAMING_LLM,
        start_size=4,
        recent_size=64,
    ),
    'KIVI': CacheConfig(
        method=BaselineMethod.KIVI,
        quantize_bits=4,
    ),
}

for name, config in baselines.items():
    print(f"\n{name}:")
    baseline = create_baseline(config)
    baseline.reset()

    # Get first layer's KV
    # DynamicCache uses different API
    k = past_key_values.key_cache[0]
    v = past_key_values.value_cache[0]

    original_len = k.shape[2]
    print(f"  Original: {k.shape}")

    # Compress
    ck, cv = baseline.compress(k, v, None, original_len - 1)
    print(f"  Compressed: {ck.shape}")
    print(f"  Ratio: {ck.shape[2] / original_len:.2%}")
    print(f"  Stats: {baseline.get_stats()}")

    # Apply to all layers
    print("  Applying to all layers...")
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        ck, cv = baseline.compress(k, v, None, original_len - 1)
        print(f"    Layer {layer_idx}: {k.shape[2]} -> {ck.shape[2]}")
        past_key_values.key_cache[layer_idx] = ck
        past_key_values.value_cache[layer_idx] = cv

    print(f"  Final cache size: {past_key_values.key_cache[0].shape[2]}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)