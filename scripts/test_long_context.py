#!/usr/bin/env python3
"""
Long Context Test with KV Cache Hooks

Tests baseline compression on real model with long context.
"""

import os
import sys

# Set HF mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import time
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


@dataclass
class LongContextResult:
    """Result from long context test."""
    method: str
    input_length: int
    output_length: int
    total_tokens: int
    cache_tokens_after_compress: int
    compression_ratio: float
    generation_time_ms: float
    perplexity: float
    generated_text: str


def load_model(model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )

    return model, tokenizer


def create_long_prompt(base_length: int = 500) -> str:
    """Create a long prompt for testing."""
    # Create repeating pattern to simulate long context
    base_text = """The development of artificial intelligence has been one of the most significant technological achievements of the 21st century. Machine learning algorithms have revolutionized how we process data, make decisions, and interact with technology. Deep learning, a subset of machine learning, has enabled breakthroughs in image recognition, natural language processing, and autonomous systems.
"""

    # Repeat to reach target length
    repetitions = base_length // len(base_text.split()) + 1
    long_prompt = base_text * repetitions

    # Add a specific question at the end
    long_prompt += "\n\nBased on the above discussion, summarize the key achievements of AI development: "

    return long_prompt


def test_full_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> LongContextResult:
    """Test generation with full cache (no compression)."""
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    print(f"  Input length: {input_length} tokens")

    # Generate
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    # Decode
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, full_text)

    return LongContextResult(
        method="Full Cache",
        input_length=input_length,
        output_length=max_new_tokens,
        total_tokens=input_length + max_new_tokens,
        cache_tokens_after_compress=input_length + max_new_tokens,
        compression_ratio=1.0,
        generation_time_ms=gen_time,
        perplexity=perplexity,
        generated_text=generated_text[:200],
    )


def test_with_compression(
    model,
    tokenizer,
    prompt: str,
    baseline,
    max_new_tokens: int = 50,
    compress_threshold: int = 300,
) -> LongContextResult:
    """Test generation with cache compression."""
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    print(f"  Input length: {input_length} tokens")

    baseline.reset()

    # Generate with manual cache management
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None
    compressions_applied = 0

    with torch.no_grad():
        # Prefill phase - compress immediately after
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # Get initial cache length using compatibility helper
        initial_cache_len = get_cache_length(past_key_values)

        # Apply initial compression (prefill compression)
        if initial_cache_len > compress_threshold:
            print(f"  Prefill compression: {initial_cache_len} tokens (threshold={compress_threshold})")
            compress_dynamic_cache(past_key_values, baseline, initial_cache_len - 1, log=True)
            new_len = get_cache_length(past_key_values)
            print(f"    Final cache: {new_len} tokens ({new_len/initial_cache_len:.1%})")
            compressions_applied += 1

        # Decode phase
        for step in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            # Get cache length using compatibility helper
            cache_len = get_cache_length(past_key_values)

            # Check compression trigger
            if cache_len >= compress_threshold:
                # Apply compression using compatibility function
                new_len = get_cache_length(past_key_values)

                compress_dynamic_cache(past_key_values, baseline, cache_len - 1, log=False)

                new_len_after = get_cache_length(past_key_values)

                if new_len_after < cache_len:
                    compressions_applied += 1
                    print(f"    Compression #{compressions_applied}: {cache_len} -> {new_len_after} tokens ({new_len_after/cache_len:.1%})")

            # Next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    # Decode
    generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, full_text)

    # Get stats
    stats = baseline.get_stats()

    # Calculate final cache length
    final_cache_len = get_cache_length(past_key_values)
    compression_ratio = final_cache_len / (input_length + max_new_tokens)

    return LongContextResult(
        method=stats.get('method', baseline.__class__.__name__.replace('Baseline', '')),
        input_length=input_length,
        output_length=max_new_tokens,
        total_tokens=input_length + max_new_tokens,
        cache_tokens_after_compress=final_cache_len,
        compression_ratio=compression_ratio,
        generation_time_ms=gen_time,
        perplexity=perplexity,
        generated_text=generated_text[:200],
    )


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity."""
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        return math.exp(loss.item()) if loss.item() < 10 else float('inf')
    except:
        return 0.0


def format_results_table(results: List[LongContextResult]) -> str:
    """Format results as table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("Long Context Baseline Comparison")
    lines.append("=" * 100)

    header = f"{'Method':<15} {'Input':>8} {'Output':>8} {'Cache':>8} {'Ratio':>10} {'Time(ms)':>12} {'PPL':>10}"
    lines.append(header)
    lines.append("-" * 100)

    for r in results:
        row = f"{r.method:<15} {r.input_length:>8} {r.output_length:>8} {r.cache_tokens_after_compress:>8} {r.compression_ratio:>10.2%} {r.generation_time_ms:>12.1f} {r.perplexity:>10.2f}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    """Main test function."""
    print("=" * 60)
    print("Long Context KV Cache Test")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model()

    # Create long prompts
    prompt_lengths = [300, 600]  # Longer prompts to trigger compression

    # Baseline configurations
    configs = {
        'Full': CacheConfig(method=BaselineMethod.FULL_CACHE),
        'H2O': CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1),
        'StreamingLLM': CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=64),
        'KIVI': CacheConfig(method=BaselineMethod.KIVI, quantize_bits=4),
    }

    # Compression thresholds
    compress_thresholds = {
        'Full': 99999,  # Never compress
        'H2O': 200,
        'StreamingLLM': 200,
        'KIVI': 200,
    }

    all_results = []

    for prompt_len in prompt_lengths:
        prompt = create_long_prompt(prompt_len)
        print(f"\n{'='*60}")
        print(f"Prompt length: ~{prompt_len} words")
        print(f"{'='*60}")

        for name, config in configs.items():
            print(f"\nTesting {name}...")

            baseline = create_baseline(config)

            if name == 'Full':
                result = test_full_cache(model, tokenizer, prompt, max_new_tokens=30)
            else:
                result = test_with_compression(
                    model, tokenizer, prompt, baseline,
                    max_new_tokens=30,
                    compress_threshold=compress_thresholds.get(name, 200),
                )

            all_results.append(result)

    # Print results
    table = format_results_table(all_results)
    print(table)

    # Print generated samples
    print("\n" + "=" * 60)
    print("Generated Text Samples")
    print("=" * 60)

    for r in all_results[:4]:
        print(f"\n{r.method} (input={r.input_length}):")
        print(f"  {r.generated_text[:150]}...")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()