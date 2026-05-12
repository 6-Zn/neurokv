#!/usr/bin/env python3
"""
Memory and Latency Profiling - Day 10

Profile memory usage and latency for NeuroKV vs baselines.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import time
import argparse
from datetime import datetime
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
from scripts.fast_policy import FastPolicyHandler


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0


def load_model(model_name='Qwen/Qwen2.5-0.5B-Instruct'):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    return model, tokenizer


def create_test_prompt(length):
    base = """The development of artificial intelligence has been one of the most significant technological achievements of the 21st century. Machine learning algorithms have revolutionized how we process data, make decisions, and interact with technology. """
    reps = length // 20 + 1
    return base * reps + "\n\nSummarize: "


class BaselineHandler:
    def __init__(self, config):
        self.baseline = create_baseline(config)
        self.stats = {'method': config.method.value}

    def reset(self):
        self.baseline.reset()

    def compress(self, cache, pos):
        compress_dynamic_cache(cache, self.baseline, pos)

    def get_stats(self):
        return self.baseline.get_stats()


def profile_generation(model, tokenizer, prompt, handler, max_tokens, threshold):
    """Profile memory and latency during generation."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    handler.reset()

    # Memory before
    mem_before = get_gpu_memory()

    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None
    compression_times = []

    with torch.no_grad():
        # Prefill
        prefill_start = time.perf_counter()
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        prefill_time = time.perf_counter() - prefill_start

        # Memory after prefill
        mem_prefill = get_gpu_memory()

        # Apply compression if needed
        compress_time = 0
        initial_len = get_cache_length(past_key_values)
        if initial_len > threshold:
            compress_start = time.perf_counter()
            if hasattr(handler, 'compress_batched'):
                handler.compress_batched(past_key_values, initial_len - 1)
            else:
                handler.compress(past_key_values, initial_len - 1)
            compress_time = time.perf_counter() - compress_start

        # Memory after compression
        mem_compress = get_gpu_memory()

        # Decode loop
        decode_times = []
        for step in range(max_tokens):
            decode_start = time.perf_counter()
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            decode_times.append(time.perf_counter() - decode_start)

            cache_len = get_cache_length(past_key_values)
            if cache_len >= threshold:
                compress_start = time.perf_counter()
                if hasattr(handler, 'compress_batched'):
                    handler.compress_batched(past_key_values, cache_len - 1)
                else:
                    handler.compress(past_key_values, cache_len - 1)
                compression_times.append(time.perf_counter() - compress_start)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    total_time = time.perf_counter() - start_time

    # Memory after generation
    mem_final = get_gpu_memory()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    # Cache size calculation
    final_cache_len = get_cache_length(past_key_values)
    layers = get_cache_layers(past_key_values)
    if layers and len(layers) > 0:
        k, v = get_layer_kv(layers[0])
        if k is not None:
            cache_bytes = k.numel() * k.element_size() * len(layers) * 2  # keys + values
            cache_mb = cache_bytes / 1024**2
        else:
            cache_mb = 0
    else:
        cache_mb = 0

    return {
        'method': handler.get_stats().get('method', handler.__class__.__name__),
        'input_tokens': input_length,
        'cache_tokens': final_cache_len,
        'prefill_time_ms': prefill_time * 1000,
        'decode_time_ms': sum(decode_times) * 1000,
        'compression_time_ms': sum(compression_times) * 1000,
        'total_time_ms': total_time * 1000,
        'mem_before_mb': mem_before[0],
        'mem_prefill_mb': mem_prefill[0],
        'mem_compress_mb': mem_compress[0],
        'mem_final_mb': mem_final[0],
        'peak_memory_mb': peak_memory,
        'cache_size_mb': cache_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory and latency profiling")
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--context-length', type=int, default=500)
    parser.add_argument('--max-tokens', type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print("Memory and Latency Profiling - Day 10")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(args.model)

    prompt = create_test_prompt(args.context_length)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"Prompt tokens: {actual_tokens}")

    threshold = max(100, actual_tokens // 3)

    handlers = {
        'Full-Cache': type('FullHandler', (), {
            'reset': lambda self: None,
            'compress': lambda self, c, p: None,
            'get_stats': lambda self: {'method': 'Full-Cache'}
        })(),
        'NeuroKV-Fast': FastPolicyHandler(),
        'H2O': BaselineHandler(CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1)),
        'StreamingLLM': BaselineHandler(CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=32)),
    }

    results = []

    print("\n" + "=" * 80)
    print("Running Profiles...")
    print("=" * 80)

    for name, handler in handlers.items():
        print(f"\n{name}...")
        result = profile_generation(model, tokenizer, prompt, handler, args.max_tokens, threshold)
        results.append(result)

    # Print results
    print("\n" + "=" * 120)
    print("Memory Profile Results")
    print("=" * 120)

    header = f"{'Method':<15} {'Input':>6} {'Cache':>6} {'CacheMB':>8} {'PeakMB':>8} {'BeforeMB':>9} {'FinalMB':>8}"
    print(header)
    print("-" * 120)

    for r in results:
        row = f"{r['method']:<15} {r['input_tokens']:>6} {r['cache_tokens']:>6} {r['cache_size_mb']:>8.1f} {r['peak_memory_mb']:>8.1f} {r['mem_before_mb']:>9.1f} {r['mem_final_mb']:>8.1f}"
        print(row)

    print("\n" + "=" * 120)
    print("Latency Profile Results")
    print("=" * 120)

    header = f"{'Method':<15} {'Prefill':>8} {'Decode':>8} {'Compress':>9} {'Total':>8}"
    print(header)
    print("-" * 120)

    for r in results:
        row = f"{r['method']:<15} {r['prefill_time_ms']:>8.1f} {r['decode_time_ms']:>8.1f} {r['compression_time_ms']:>9.1f} {r['total_time_ms']:>8.1f}"
        print(row)

    print("=" * 120)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    full_result = results[0]
    neurokv_result = results[1]

    cache_reduction = (full_result['cache_tokens'] - neurokv_result['cache_tokens']) / full_result['cache_tokens']
    memory_saved = full_result['cache_size_mb'] - neurokv_result['cache_size_mb']

    print(f"\nNeuroKV vs Full-Cache:")
    print(f"  Cache reduction: {cache_reduction:.1%}")
    print(f"  Memory saved: {memory_saved:.1f} MB")
    print(f"  Compression overhead: {neurokv_result['compression_time_ms']:.1f} ms")

    print("\n" + "=" * 60)
    print("Profiling Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()