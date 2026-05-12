#!/usr/bin/env python3
"""
NeuroKV Benchmark Evaluation - Day 8

Comprehensive evaluation on multiple test scenarios:
1. Variable context lengths (100, 300, 500, 800 tokens)
2. Different generation lengths
3. Multiple threshold settings
4. Quality metrics: perplexity, coherence, compression ratio
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import time
import math
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict
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


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    context_length: int
    method: str
    input_tokens: int
    cache_tokens: int
    compression_ratio: float
    generation_time_ms: float
    perplexity: float
    coherence_score: float  # 0-1 based on output quality
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


def create_test_prompts() -> Dict[str, str]:
    """Create diverse test prompts."""
    prompts = {
        'short_qa': """What is machine learning? Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. """,

        'medium_summarize': """The Amazon rainforest covers about 5.5 million square kilometers and is home to an estimated 400 billion trees. It spans nine countries in South America, with the majority in Brazil. The rainforest plays a crucial role in regulating the global climate and is often referred to as the "lungs of the Earth."

However, deforestation has become a major concern. In recent decades, significant portions of the rainforest have been cleared for agriculture, logging, and urban development. This has led to biodiversity loss, climate change acceleration, and displacement of indigenous communities.

Scientists estimate that the Amazon stores about 150-200 billion tons of carbon, making its preservation critical for climate stability. Conservation efforts include establishing protected areas, promoting sustainable land use, and supporting indigenous rights.

Summarize the key environmental importance of the Amazon rainforest: """,

        'long_context': """Artificial intelligence has evolved significantly over the past decade, with breakthroughs in natural language processing, computer vision, and reinforcement learning. Large language models like GPT and BERT have demonstrated remarkable capabilities in understanding and generating human-like text. These models use transformer architectures with billions of parameters trained on vast datasets.

In parallel, computer vision has advanced through convolutional neural networks and more recently, vision transformers. Applications range from autonomous vehicles to medical imaging diagnosis. Reinforcement learning has shown success in game-playing AI like AlphaGo and robotic control systems.

The integration of these technologies has led to multimodal AI systems that can process text, images, and audio together. This convergence enables applications like visual question answering, image captioning, and video understanding.

However, challenges remain. Large models require significant computational resources, raising concerns about energy consumption and accessibility. Bias in training data can lead to unfair or harmful outputs. Privacy and security issues need careful consideration when deploying AI in sensitive domains.

Research directions include making models more efficient through techniques like quantization, pruning, and knowledge distillation. Federated learning addresses privacy concerns by training on distributed data. Interpretability research aims to make AI decisions more transparent and trustworthy.

Based on the above discussion, what are the key challenges and future research directions in AI? """,

        'mathematical': """Given a sequence: 2, 4, 8, 16, 32, what is the next number? The sequence follows a pattern where each number is twice the previous one. So the next number would be 64. """,

        'code_context': """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

This is a classic recursive sorting algorithm with average time complexity O(n log n) but worst case O(n²) when the pivot selection is poor. """,
    }
    return prompts


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


def compute_coherence_score(text: str) -> float:
    """
    Compute simple coherence score based on:
    - Non-repetitive content
    - Presence of actual words (not numbers/garbage)
    - Reasonable length
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    # Check for repetitive patterns
    words = text.split()
    if len(words) < 5:
        return 0.1

    # Check word diversity
    unique_words = set(words)
    diversity = len(unique_words) / len(words)

    # Check for garbled content (many non-ASCII or numbers)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)

    # Check for repetitive sequences like "0 0 0 0"
    if len(words) > 10:
        consecutive_same = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_same += 1
        repetition_penalty = min(1.0, consecutive_same / len(words))
    else:
        repetition_penalty = 0

    # Combined score
    score = diversity * 0.4 + alpha_ratio * 0.4 + (1 - repetition_penalty) * 0.2
    return max(0.0, min(1.0, score))


class NeuroKVPolicyEvaluator:
    """Evaluator for NeuroKV policy."""

    def __init__(self, policy_path: str, device: str = 'cuda'):
        checkpoint = torch.load(policy_path, map_location=device)
        self.config = checkpoint['config']

        from scripts.train_binary_policy import BinaryPolicyNetwork
        self.policy = BinaryPolicyNetwork(
            hidden_size=self.config['hidden_size'],
            num_llm_layers=self.config['num_llm_layers'],
            feature_dim=self.config['feature_dim'],
        ).to(device)
        self.policy.load_state_dict(checkpoint['model'])
        self.policy.eval()

        self.device = device
        self.keep_ratio = 0.15
        self.stats = {'method': 'NeuroKV-Policy', 'compressions': 0}

    def reset(self):
        self.stats['compressions'] = 0

    def compress(self, cache, current_position: int):
        """Apply policy compression."""
        layers = get_cache_layers(cache)
        if layers is None:
            return

        for layer_idx, layer in enumerate(layers):
            keys, values = get_layer_kv(layer)
            if keys is None:
                continue

            seq_len = keys.shape[2]

            # Always keep sinks and recent
            sink_size = 4
            recent_size = max(16, int(seq_len * 0.05))

            eval_start = sink_size
            eval_end = seq_len - recent_size

            if eval_end <= eval_start:
                continue

            features = self._extract_features(keys, values, seq_len, layer_idx, current_position)
            positions = torch.arange(seq_len, device=self.device).clamp_max(131071)
            layer_ids = torch.full((seq_len,), layer_idx, device=self.device, dtype=torch.long)

            with torch.no_grad():
                features_batch = features.unsqueeze(0)
                positions_batch = positions.unsqueeze(0)
                logits = self.policy(features_batch, positions_batch, layer_ids.unsqueeze(0))
                predictions = logits.squeeze(0).argmax(dim=-1)

            sink_indices = torch.arange(0, sink_size, device=self.device)
            recent_indices = torch.arange(seq_len - recent_size, seq_len, device=self.device)

            middle_predictions = predictions[eval_start:eval_end]
            middle_keep = (middle_predictions == 0).nonzero(as_tuple=True)[0] + eval_start

            keep_indices = torch.unique(torch.cat([sink_indices, middle_keep, recent_indices]))

            min_keep = max(sink_size + recent_size, int(seq_len * self.keep_ratio))
            if len(keep_indices) < min_keep:
                additional = min_keep - len(keep_indices)
                extra = torch.arange(seq_len - recent_size - additional, seq_len - recent_size, device=self.device)
                keep_indices = torch.unique(torch.cat([keep_indices, extra]))

            if len(keep_indices) < seq_len:
                new_keys = keys[:, :, keep_indices, :]
                new_values = values[:, :, keep_indices, :]
                set_layer_kv(layer, new_keys, new_values)

        self.stats['compressions'] += 1

    def _extract_features(self, keys, values, seq_len, layer_idx, current_position):
        features = torch.zeros(seq_len, self.config['feature_dim'], device=self.device)
        features[:, 0] = torch.arange(seq_len, device=self.device).float() / 10000
        features[:, 1] = (current_position - torch.arange(seq_len, device=self.device)).float() / 100
        features[:, 2] = layer_idx / self.config['num_llm_layers']

        if keys is not None:
            key_norm = keys[0].norm(dim=-1).mean(dim=0)
            value_norm = values[0].norm(dim=-1).mean(dim=0)
            features[:, 3] = key_norm / (key_norm.max() + 1e-8)
            features[:, 4] = value_norm / (value_norm.max() + 1e-8)

        features[:, 5] = torch.log1p(torch.clamp(current_position - torch.arange(seq_len, device=self.device), min=0).float())
        return features

    def get_stats(self):
        return self.stats


def run_generation_with_compression(model, tokenizer, prompt, compression_handler, max_new_tokens, threshold):
    """Run generation with compression handler."""
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    compression_handler.reset()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        initial_len = get_cache_length(past_key_values)

        if initial_len > threshold:
            compression_handler.compress(past_key_values, initial_len - 1)

        for step in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            cache_len = get_cache_length(past_key_values)
            if cache_len >= threshold:
                compression_handler.compress(past_key_values, cache_len - 1)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    perplexity = compute_perplexity(model, tokenizer, full_text)
    coherence = compute_coherence_score(generated_text)
    final_cache = get_cache_length(past_key_values)

    return {
        'input_tokens': input_length,
        'cache_tokens': final_cache,
        'compression_ratio': final_cache / (input_length + max_new_tokens),
        'generation_time_ms': gen_time,
        'perplexity': perplexity,
        'coherence': coherence,
        'generated_text': generated_text[:200],
    }


def run_benchmark(model, tokenizer, methods, prompts, max_new_tokens=30, thresholds_by_length=None):
    """Run comprehensive benchmark."""
    results = []

    if thresholds_by_length is None:
        thresholds_by_length = {
            'short': 100,
            'medium': 200,
            'long': 300,
            'very_long': 400,
        }

    for prompt_name, prompt in prompts.items():
        input_len = len(tokenizer.encode(prompt))
        length_category = 'short' if input_len < 100 else 'medium' if input_len < 300 else 'long' if input_len < 600 else 'very_long'
        threshold = thresholds_by_length.get(length_category, 200)

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_name} ({input_len} tokens, threshold={threshold})")
        print(f"{'='*60}")

        for method_name, handler in methods.items():
            print(f"  Testing {method_name}...")

            result_data = run_generation_with_compression(
                model, tokenizer, prompt, handler, max_new_tokens, threshold
            )

            result = BenchmarkResult(
                context_length=input_len,
                method=method_name,
                input_tokens=result_data['input_tokens'],
                cache_tokens=result_data['cache_tokens'],
                compression_ratio=result_data['compression_ratio'],
                generation_time_ms=result_data['generation_time_ms'],
                perplexity=result_data['perplexity'],
                coherence_score=result_data['coherence'],
                generated_text=result_data['generated_text'],
            )
            results.append(result)

            print(f"    Cache: {result.cache_tokens} ({result.compression_ratio:.1%}), PPL: {result.perplexity:.2f}, Coherence: {result.coherence_score:.2f}")

    return results


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format all results as table."""
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("NeuroKV Benchmark Evaluation Results")
    lines.append("=" * 120)

    header = f"{'Context':>8} {'Method':<18} {'Input':>6} {'Cache':>6} {'Ratio':>8} {'Time':>8} {'PPL':>6} {'Coh':>6}"
    lines.append(header)
    lines.append("-" * 120)

    for r in results:
        row = f"{r.context_length:>8} {r.method:<18} {r.input_tokens:>6} {r.cache_tokens:>6} {r.compression_ratio:>8.1%} {r.generation_time_ms:>8.1f} {r.perplexity:>6.2f} {r.coherence_score:>6.2f}"
        lines.append(row)

    lines.append("=" * 120)
    return "\n".join(lines)


def compute_summary_stats(results: List[BenchmarkResult]) -> Dict:
    """Compute summary statistics per method."""
    stats = {}

    methods = set(r.method for r in results)

    for method in methods:
        method_results = [r for r in results if r.method == method]

        stats[method] = {
            'avg_compression': sum(r.compression_ratio for r in method_results) / len(method_results),
            'avg_perplexity': sum(r.perplexity for r in method_results) / len(method_results),
            'avg_coherence': sum(r.coherence_score for r in method_results) / len(method_results),
            'avg_time': sum(r.generation_time_ms for r in method_results) / len(method_results),
            'count': len(method_results),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="NeuroKV Benchmark Evaluation")
    parser.add_argument('--policy', type=str, default='checkpoints/binary_policy.pt')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max-tokens', type=int, default=30)
    parser.add_argument('--output', type=str, default='results/benchmark_results.json')
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroKV Benchmark Evaluation - Day 8")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Create test prompts
    prompts = create_test_prompts()

    # Initialize methods
    methods = {}

    # NeuroKV Policy
    try:
        policy = NeuroKVPolicyEvaluator(args.policy)
        methods['NeuroKV-Policy'] = policy
    except Exception as e:
        print(f"Warning: Could not load policy: {e}")

    # Baselines
    baseline_configs = {
        'H2O': CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1),
        'StreamingLLM': CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=32),
        'Full-Cache': CacheConfig(method=BaselineMethod.FULL_CACHE),
    }

    class FullCacheHandler:
        def __init__(self):
            self.stats = {'method': 'Full-Cache'}
        def reset(self):
            pass
        def compress(self, cache, pos):
            pass  # No compression
        def get_stats(self):
            return self.stats

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

    methods['Full-Cache'] = FullCacheHandler()

    for name, config in baseline_configs.items():
        if name != 'Full-Cache':
            methods[name] = BaselineHandler(config)

    # Run benchmark
    results = run_benchmark(model, tokenizer, methods, prompts, args.max_tokens)

    # Print results
    table = format_results_table(results)
    print(table)

    # Compute summary
    stats = compute_summary_stats(results)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for method, s in stats.items():
        print(f"\n{method}:")
        print(f"  Avg Compression: {s['avg_compression']:.1%}")
        print(f"  Avg Perplexity: {s['avg_perplexity']:.2f}")
        print(f"  Avg Coherence: {s['avg_coherence']:.2f}")
        print(f"  Avg Time: {s['avg_time']:.1f}ms")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'max_tokens': args.max_tokens,
        'results': [asdict(r) for r in results],
        'summary': stats,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()