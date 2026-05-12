#!/usr/bin/env python3
"""
NeuroKV Policy Evaluation - Day 7

Evaluate trained policy network against baseline methods on actual KV cache compression.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict, List

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
class EvalResult:
    """Result from policy evaluation."""
    method: str
    input_length: int
    output_length: int
    cache_tokens: int
    compression_ratio: float
    generation_time_ms: float
    perplexity: float
    generated_text: str


class NeuroKVPolicyBaseline:
    """
    NeuroKV policy integrated as a baseline method.

    Uses trained binary policy to decide which tokens to keep.
    """

    def __init__(self, policy_path: str, device: str = 'cuda'):
        # Load policy
        checkpoint = torch.load(policy_path, map_location=device)
        self.config = checkpoint['config']

        # Import and create model
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

        # Stats
        self.stats = {'method': 'NeuroKV-Policy', 'compressions': 0}

    def reset(self):
        self.stats['compressions'] = 0

    def compress(self, cache, current_position: int, importance_scores: torch.Tensor = None):
        """
        Apply policy to compress KV cache.

        Args:
            cache: DynamicCache or tuple cache
            current_position: Current generation position
            importance_scores: Optional importance scores (for feature extraction)
        """
        layers = get_cache_layers(cache)
        if layers is None:
            return

        total_evicted = 0

        for layer_idx, layer in enumerate(layers):
            keys, values = get_layer_kv(layer)
            if keys is None:
                continue

            seq_len = keys.shape[2]

            # Always keep attention sink tokens (first 4) and recent tokens
            sink_size = 4
            recent_size = max(16, int(seq_len * 0.05))

            # Extract features for positions that can be evaluated
            # Skip sink tokens and recent tokens - they should always be kept
            eval_start = sink_size
            eval_end = seq_len - recent_size

            if eval_end <= eval_start:
                # Sequence too short, keep all
                continue

            # Extract features for evaluable positions
            features = self._extract_features(keys, values, seq_len, layer_idx, current_position)

            # Get policy predictions for middle positions only
            positions = torch.arange(seq_len, device=self.device).clamp_max(131071)
            layer_ids = torch.full((seq_len,), layer_idx, device=self.device, dtype=torch.long)

            with torch.no_grad():
                features_batch = features.unsqueeze(0)
                positions_batch = positions.unsqueeze(0)
                logits = self.policy(features_batch, positions_batch, layer_ids.unsqueeze(0))
                predictions = logits.squeeze(0).argmax(dim=-1)

            # Build keep indices: sinks + policy-kept middle + recent
            sink_indices = torch.arange(0, sink_size, device=self.device)
            recent_indices = torch.arange(seq_len - recent_size, seq_len, device=self.device)

            # From middle region, keep those predicted as KEEP (0)
            middle_predictions = predictions[eval_start:eval_end]
            middle_keep = (middle_predictions == 0).nonzero(as_tuple=True)[0] + eval_start

            # Combine
            keep_indices = torch.unique(torch.cat([sink_indices, middle_keep, recent_indices]))

            # Ensure minimum keep ratio
            min_keep = max(sink_size + recent_size, int(seq_len * self.keep_ratio))
            if len(keep_indices) < min_keep:
                # Add more from recent region
                additional_needed = min_keep - len(keep_indices)
                extra_recent = torch.arange(seq_len - recent_size - additional_needed, seq_len - recent_size, device=self.device)
                keep_indices = torch.unique(torch.cat([keep_indices, extra_recent]))

            # Keep selected tokens
            if len(keep_indices) < seq_len:
                new_keys = keys[:, :, keep_indices, :]
                new_values = values[:, :, keep_indices, :]
                set_layer_kv(layer, new_keys, new_values)
                total_evicted += seq_len - len(keep_indices)

        self.stats['compressions'] += 1
        return total_evicted

    def _extract_features(self, keys, values, seq_len, layer_idx, current_position):
        """Extract features for policy input."""
        # Simplified features based on KV cache characteristics
        features = torch.zeros(seq_len, self.config['feature_dim'], device=self.device)

        # Position features
        features[:, 0] = torch.arange(seq_len, device=self.device).float() / 10000
        features[:, 1] = (current_position - torch.arange(seq_len, device=self.device)).float() / 100

        # Layer feature
        features[:, 2] = layer_idx / self.config['num_llm_layers']

        # KV magnitude features (approximation of importance)
        if keys is not None:
            key_norm = keys[0].norm(dim=-1).mean(dim=0)  # (seq_len,)
            value_norm = values[0].norm(dim=-1).mean(dim=0)  # (seq_len,)
            features[:, 3] = key_norm / key_norm.max()
            features[:, 4] = value_norm / value_norm.max()

        # Relative position to current
        features[:, 5] = torch.log1p((current_position - torch.arange(seq_len, device=self.device)).float())

        # Placeholder for additional features
        # In full implementation, would use attention history

        return features

    def get_stats(self):
        return self.stats


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


def create_test_prompt(length: int = 300) -> str:
    """Create test prompt."""
    base = """The development of artificial intelligence has been one of the most significant technological achievements of the 21st century. Machine learning algorithms have revolutionized how we process data, make decisions, and interact with technology. Deep learning, a subset of machine learning, has enabled breakthroughs in image recognition, natural language processing, and autonomous systems.
"""
    reps = length // len(base.split()) + 1
    prompt = base * reps + "\n\nBased on the above discussion, summarize the key achievements of AI development: "
    return prompt


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


def test_with_policy(model, tokenizer, prompt: str, policy, max_new_tokens: int = 50, threshold: int = 200):
    """Test generation with policy-based compression."""
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    print(f"  Input length: {input_length} tokens")

    policy.reset()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None
    compressions = 0

    with torch.no_grad():
        # Prefill
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        initial_len = get_cache_length(past_key_values)

        # Apply initial compression
        if initial_len > threshold:
            print(f"  Prefill compression: {initial_len} -> applying policy")
            policy.compress(past_key_values, initial_len - 1)
            new_len = get_cache_length(past_key_values)
            print(f"    After policy: {new_len} tokens ({new_len/initial_len:.1%})")
            compressions += 1

        # Decode
        for step in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            cache_len = get_cache_length(past_key_values)

            if cache_len >= threshold:
                policy.compress(past_key_values, cache_len - 1)
                compressions += 1

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    perplexity = compute_perplexity(model, tokenizer, full_text)

    final_cache = get_cache_length(past_key_values)

    return EvalResult(
        method=policy.stats['method'],
        input_length=input_length,
        output_length=max_new_tokens,
        cache_tokens=final_cache,
        compression_ratio=final_cache / (input_length + max_new_tokens),
        generation_time_ms=gen_time,
        perplexity=perplexity,
        generated_text=generated_text[:200],
    )


def test_with_baseline(model, tokenizer, prompt: str, baseline, max_new_tokens: int = 50, threshold: int = 200):
    """Test with baseline compression."""
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    input_length = input_ids.shape[1]

    print(f"  Input length: {input_length} tokens")

    baseline.reset()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        initial_len = get_cache_length(past_key_values)

        if initial_len > threshold:
            compress_dynamic_cache(past_key_values, baseline, initial_len - 1)
            new_len = get_cache_length(past_key_values)
            print(f"    Baseline compression: {initial_len} -> {new_len}")

        for step in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            cache_len = get_cache_length(past_key_values)

            if cache_len >= threshold:
                compress_dynamic_cache(past_key_values, baseline, cache_len - 1)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    perplexity = compute_perplexity(model, tokenizer, full_text)

    final_cache = get_cache_length(past_key_values)

    return EvalResult(
        method=baseline.get_stats().get('method', baseline.__class__.__name__.replace('Baseline', '')),
        input_length=input_length,
        output_length=max_new_tokens,
        cache_tokens=final_cache,
        compression_ratio=final_cache / (input_length + max_new_tokens),
        generation_time_ms=gen_time,
        perplexity=perplexity,
        generated_text=generated_text[:200],
    )


def format_table(results: List[EvalResult]) -> str:
    """Format results as table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("NeuroKV Policy Evaluation Results")
    lines.append("=" * 100)
    header = f"{'Method':<20} {'Input':>8} {'Cache':>8} {'Ratio':>10} {'Time':>10} {'PPL':>10}"
    lines.append(header)
    lines.append("-" * 100)
    for r in results:
        row = f"{r.method:<20} {r.input_length:>8} {r.cache_tokens:>8} {r.compression_ratio:>10.2%} {r.generation_time_ms:>10.1f} {r.perplexity:>10.2f}"
        lines.append(row)
    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuroKV policy")
    parser.add_argument('--policy', type=str, default='checkpoints/binary_policy.pt')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--prompt-length', type=int, default=300)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--threshold', type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroKV Policy Evaluation - Day 7")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Create test prompt
    prompt = create_test_prompt(args.prompt_length)

    # Initialize methods
    results = []

    # Test NeuroKV Policy
    print("\n" + "-" * 40)
    print("Testing NeuroKV Policy...")
    print("-" * 40)

    try:
        policy = NeuroKVPolicyBaseline(args.policy)
        result = test_with_policy(model, tokenizer, prompt, policy, args.max_tokens, args.threshold)
        results.append(result)
    except Exception as e:
        print(f"  Policy test failed: {e}")

    # Test baselines
    baselines = {
        'H2O': CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1),
        'StreamingLLM': CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=32),
    }

    for name, config in baselines.items():
        print(f"\n" + "-" * 40)
        print(f"Testing {name}...")
        print("-" * 40)

        baseline = create_baseline(config)
        result = test_with_baseline(model, tokenizer, prompt, baseline, args.max_tokens, args.threshold)
        results.append(result)

    # Print table
    table = format_table(results)
    print(table)

    # Print generated samples
    print("\nGenerated Text Samples:")
    for r in results:
        print(f"\n{r.method}:")
        print(f"  {r.generated_text[:100]}...")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()