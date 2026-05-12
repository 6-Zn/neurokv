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
    get_cache_layers,
    get_layer_kv,
    set_layer_kv,
    get_cache_length,
    compress_dynamic_cache,
)

print("=" * 60)
print("Direct Compression Test (transformers 5.8+ Compatible)")
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

# Check cache structure using compatibility helpers
print(f"Past key values type: {type(past_key_values).__name__}")
cache_len = get_cache_length(past_key_values)
print(f"Cache length: {cache_len} tokens")

layers = get_cache_layers(past_key_values)
print(f"Number of layers: {len(layers) if layers else 0}")

if layers and len(layers) > 0:
    k0, v0 = get_layer_kv(layers[0])
    if k0 is not None:
        print(f"Layer 0 keys shape: {k0.shape}")

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

    # Get original length
    original_len = get_cache_length(past_key_values)
    print(f"  Original cache length: {original_len}")

    # Compress using compatibility function
    compress_dynamic_cache(past_key_values, baseline, original_len - 1, log=True)

    # Get new length
    new_len = get_cache_length(past_key_values)
    print(f"  Final cache length: {new_len}")
    print(f"  Compression ratio: {new_len / original_len:.2%}")
    print(f"  Stats: {baseline.get_stats()}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)