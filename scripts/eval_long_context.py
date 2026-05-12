#!/usr/bin/env python3
"""
Long Context Evaluation - Day 9

Test NeuroKV on 1000+ token contexts.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import time
import math
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


def create_long_prompt(target_tokens: int) -> str:
    """Create prompt with target token count."""
    base_text = """The history of computing spans centuries, from ancient calculating devices to modern quantum computers. The abacus, developed around 2400 BC, was one of the earliest calculation tools. Mechanical calculators like Pascal's calculator (1642) and Babbage's Difference Engine (1822) laid groundwork for automatic computation.

The electronic computer era began in the 1940s with machines like ENIAC and UNIVAC. These massive machines used vacuum tubes and could perform thousands of calculations per second. The invention of the transistor in 1947 and integrated circuits in 1958 enabled smaller, faster, and more reliable computers.

Personal computers emerged in the 1970s with Apple II and IBM PC. The graphical user interface, introduced by Macintosh in 1984, made computers accessible to non-technical users. The internet revolution of the 1990s connected billions of devices worldwide.

Cloud computing transformed infrastructure in the 2000s, allowing businesses to rent computing resources. Mobile computing put powerful computers in pockets. Artificial intelligence and machine learning became mainstream technologies.

Quantum computing represents the next frontier. Quantum computers use quantum mechanics to solve problems intractable for classical computers. Companies like IBM, Google, and startups race to achieve quantum advantage.

Cybersecurity evolved from simple passwords to sophisticated threat detection. Blockchain technology introduced decentralized, tamper-resistant record keeping. Virtual and augmented reality create immersive digital experiences.

Edge computing moves processing closer to data sources. Neuromorphic computing mimics brain architecture. Biocomputing uses DNA and proteins for computation. The field continues advancing rapidly.

"""
    # Calculate repetitions needed
    words_per_block = len(base_text.split())
    blocks_needed = target_tokens // words_per_block + 1
    prompt = base_text * blocks_needed

    # Add question at end
    prompt += "\n\nSummarize the key milestones in computing history from ancient times to quantum computing: "

    return prompt


def compute_perplexity(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return math.exp(loss.item()) if loss.item() < 10 else float('inf')
    except:
        return 0.0


def compute_coherence(text):
    if not text or len(text.strip()) < 10:
        return 0.0
    words = text.split()
    if len(words) < 5:
        return 0.1
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    consecutive_same = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
    repetition_penalty = min(1.0, consecutive_same / len(words))
    score = diversity * 0.4 + alpha_ratio * 0.4 + (1 - repetition_penalty) * 0.2
    return max(0.0, min(1.0, score))


def test_with_handler(model, tokenizer, prompt, handler, max_tokens, threshold):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    print(f"  Input: {input_length} tokens, threshold: {threshold}")

    handler.reset()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        # Prefill
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        initial_len = get_cache_length(past_key_values)

        if initial_len > threshold:
            if hasattr(handler, 'compress_batched'):
                handler.compress_batched(past_key_values, initial_len - 1)
            else:
                handler.compress(past_key_values, initial_len - 1)
            new_len = get_cache_length(past_key_values)
            print(f"    Prefill: {initial_len} -> {new_len} ({new_len/initial_len:.1%})")

        # Decode
        for step in range(max_tokens):
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            cache_len = get_cache_length(past_key_values)
            if cache_len >= threshold:
                if hasattr(handler, 'compress_batched'):
                    handler.compress_batched(past_key_values, cache_len - 1)
                else:
                    handler.compress(past_key_values, cache_len - 1)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    perplexity = compute_perplexity(model, tokenizer, full_text)
    coherence = compute_coherence(generated_text)
    final_cache = get_cache_length(past_key_values)

    stats = handler.get_stats()
    inference_time = stats.get('inference_time_ms', 0)

    return {
        'method': stats.get('method', handler.__class__.__name__),
        'input_tokens': input_length,
        'cache_tokens': final_cache,
        'compression_ratio': final_cache / (input_length + max_tokens),
        'gen_time_ms': gen_time,
        'policy_time_ms': inference_time,
        'perplexity': perplexity,
        'coherence': coherence,
        'generated_text': generated_text[:150],
    }


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


def main():
    parser = argparse.ArgumentParser(description="Long context evaluation")
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--context-length', type=int, default=1000)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--threshold', type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("Long Context Evaluation - Day 9")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target context: {args.context_length} tokens")

    model, tokenizer = load_model(args.model)

    prompt = create_long_prompt(args.context_length)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"Actual prompt tokens: {actual_tokens}")

    # Adjust threshold if needed
    threshold = args.threshold
    if actual_tokens < threshold:
        threshold = max(100, actual_tokens // 3)
        print(f"Adjusted threshold to {threshold}")

    # Initialize handlers
    handlers = {
        'NeuroKV-Fast': FastPolicyHandler(),
        'H2O': BaselineHandler(CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1)),
        'StreamingLLM': BaselineHandler(CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=64)),
    }

    results = []

    print("\n" + "=" * 60)
    print("Running Tests...")
    print("=" * 60)

    for name, handler in handlers.items():
        print(f"\n{name}:")
        result = test_with_handler(model, tokenizer, prompt, handler, args.max_tokens, threshold)
        results.append(result)

    # Print results table
    print("\n" + "=" * 100)
    print("Results")
    print("=" * 100)

    header = f"{'Method':<20} {'Input':>6} {'Cache':>6} {'Ratio':>8} {'GenTime':>8} {'Policy':>8} {'PPL':>6} {'Coh':>6}"
    print(header)
    print("-" * 100)

    for r in results:
        row = f"{r['method']:<20} {r['input_tokens']:>6} {r['cache_tokens']:>6} {r['compression_ratio']:>8.1%} {r['gen_time_ms']:>8.1f} {r['policy_time_ms']:>8.1f} {r['perplexity']:>6.2f} {r['coherence']:>6.2f}"
        print(row)

    print("=" * 100)

    # Print generated samples
    print("\nGenerated Text Samples:")
    for r in results:
        print(f"\n{r['method']}:")
        print(f"  {r['generated_text'][:100]}...")

    print("\n" + "=" * 60)
    print("Long Context Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()