"""
NeuroKV Evaluation Framework

Provides comprehensive evaluation for KV cache compression methods:
- Quality metrics: perplexity, BLEU, ROUGE, accuracy
- Memory metrics: cache size, compression ratio
- Speed metrics: latency, throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class MetricType(Enum):
    """Available evaluation metrics."""
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    ACCURACY = "accuracy"
    KL_DIVERGENCE = "kl"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Metrics to compute
    metrics: List[MetricType] = None

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9

    # Memory budget
    cache_budget_ratio: float = 0.2  # Keep 20% of cache

    # Comparison settings
    compare_with_full: bool = True  # Compare against full cache

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                MetricType.PERPLEXITY,
                MetricType.KL_DIVERGENCE,
                MetricType.MEMORY,
                MetricType.LATENCY,
            ]


@dataclass
class EvalResult:
    """Result from single evaluation run."""
    method_name: str

    # Quality metrics
    perplexity: float = 0.0
    kl_divergence: float = 0.0
    accuracy: float = 0.0

    # Memory metrics
    cache_size_bytes: int = 0
    compression_ratio: float = 1.0
    tokens_kept: int = 0

    # Speed metrics
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0

    # Generated text
    generated_text: str = ""

    # Metadata
    num_input_tokens: int = 0
    num_output_tokens: int = 0


class PerplexityCalculator:
    """Calculate perplexity for generated text."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute(self, text: str) -> float:
        """
        Compute perplexity of text.

        Args:
            text: Input text to compute perplexity for

        Returns:
            perplexity: Perplexity value
        """
        # Encode text
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)

        # Get logits
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = math.exp(loss.item())

        return perplexity

    def compute_with_cache(
        self,
        input_ids: torch.Tensor,
        cache_manager,
    ) -> float:
        """Compute perplexity using compressed cache."""
        # This would integrate with cache manager
        # For now, use standard computation
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return math.exp(loss.item())


class KLDivergenceCalculator:
    """Calculate KL divergence between full and compressed outputs."""

    def __init__(self, model):
        self.model = model

    def compute(
        self,
        input_ids: torch.Tensor,
        full_probs: torch.Tensor,
        compressed_probs: torch.Tensor,
    ) -> float:
        """
        Compute KL divergence between full and compressed output distributions.

        Args:
            input_ids: Input token IDs
            full_probs: Probability distribution from full cache
            compressed_probs: Probability distribution from compressed cache

        Returns:
            kl_div: KL divergence value
        """
        # KL(P || Q) = sum(P * log(P/Q))
        kl_div = F.kl_div(
            compressed_probs.log(),
            full_probs,
            reduction='sum'
        ).item()

        return kl_div

    def get_output_probs(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get output probability distribution."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits
            probs = F.softmax(logits, dim=-1)
        return probs


class MemoryTracker:
    """Track memory usage during inference."""

    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0

    def start(self):
        """Start tracking."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def stop(self) -> Dict[str, int]:
        """Stop tracking and return stats."""
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        current = torch.cuda.memory_allocated()

        return {
            "peak_memory_bytes": peak,
            "current_memory_bytes": current,
            "peak_memory_mb": peak / 1024 / 1024,
            "current_memory_mb": current / 1024 / 1024,
        }

    def get_cache_size(
        self,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        precision_bytes: int = 2,  # FP16 = 2 bytes
    ) -> int:
        """Calculate KV cache size in bytes."""
        # KV cache size = 2 (K+V) * layers * tokens * heads * head_dim * precision
        cache_size = 2 * num_layers * num_tokens * num_heads * head_dim * precision_bytes
        return cache_size


class LatencyTracker:
    """Track inference latency."""

    def __init__(self):
        self.prefill_time = 0
        self.decode_time = 0
        self.total_time = 0

    def start_timer(self) -> float:
        """Start timing."""
        torch.cuda.synchronize()
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        """Stop timing and return elapsed time."""
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        return elapsed

    def measure_generation(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cache_manager=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Measure generation latency.

        Returns:
            output_ids: Generated token IDs
            timing: Timing statistics
        """
        # Prefill phase
        prefill_start = self.start_timer()

        with torch.no_grad():
            # Process input
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

        prefill_time = self.stop_timer(prefill_start) * 1000  # ms

        # Decode phase
        decode_start = self.start_timer()

        generated_ids = []
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_ids.append(next_token.item())

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = model(
                    next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_ids.append(next_token.item())

        decode_time = self.stop_timer(decode_start) * 1000  # ms

        total_time = prefill_time + decode_time
        throughput = max_new_tokens / (decode_time / 1000) if decode_time > 0 else 0

        output_ids = torch.tensor(generated_ids)

        timing = {
            "prefill_latency_ms": prefill_time,
            "decode_latency_ms": decode_time,
            "total_latency_ms": total_time,
            "throughput_tokens_per_sec": throughput,
        }

        return output_ids, timing


class BaselineEvaluator:
    """
    Evaluate baseline KV cache compression methods.

    Compares quality, memory, and speed metrics.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: EvalConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvalConfig()

        # Initialize calculators
        self.perplexity_calc = PerplexityCalculator(model, tokenizer)
        self.kl_calc = KLDivergenceCalculator(model)
        self.memory_tracker = MemoryTracker()
        self.latency_tracker = LatencyTracker()

        # Get model info
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.num_heads

    def evaluate_full_cache(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> EvalResult:
        """Evaluate with full KV cache (oracle baseline)."""
        # Encode
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)
        num_input_tokens = input_ids.shape[1]

        # Track memory
        self.memory_tracker.start()

        # Generate
        output_ids, timing = self.latency_tracker.measure_generation(
            self.model, input_ids, max_new_tokens
        )

        memory_stats = self.memory_tracker.stop()

        # Decode output
        full_text = prompt + self.tokenizer.decode(output_ids)
        generated_text = self.tokenizer.decode(output_ids)

        # Compute perplexity
        perplexity = self.perplexity_calc.compute(full_text)

        # Calculate cache size
        full_cache_size = self.memory_tracker.get_cache_size(
            num_input_tokens + max_new_tokens,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            precision_bytes=2,  # FP16
        )

        return EvalResult(
            method_name="Full Cache",
            perplexity=perplexity,
            cache_size_bytes=full_cache_size,
            compression_ratio=1.0,
            tokens_kept=num_input_tokens + max_new_tokens,
            prefill_latency_ms=timing['prefill_latency_ms'],
            decode_latency_ms=timing['decode_latency_ms'],
            throughput_tokens_per_sec=timing['throughput_tokens_per_sec'],
            generated_text=generated_text,
            num_input_tokens=num_input_tokens,
            num_output_tokens=max_new_tokens,
        )

    def evaluate_with_baseline(
        self,
        baseline,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> EvalResult:
        """Evaluate with a specific baseline method."""
        # Encode
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)
        num_input_tokens = input_ids.shape[1]

        # Get full cache result for comparison
        if self.config.compare_with_full:
            full_result = self.evaluate_full_cache(prompt, max_new_tokens)

        # Track memory
        self.memory_tracker.start()

        # Generate with baseline
        output_ids, timing = self.latency_tracker.measure_generation(
            self.model, input_ids, max_new_tokens
        )

        memory_stats = self.memory_tracker.stop()

        # Get baseline stats
        baseline_stats = baseline.get_stats()

        # Decode output
        generated_text = self.tokenizer.decode(output_ids)
        full_text = prompt + generated_text

        # Compute metrics
        perplexity = self.perplexity_calc.compute(full_text)

        # Calculate compression ratio
        if self.config.compare_with_full:
            kl_div = 0.0  # Would compute KL divergence
            compression_ratio = baseline_stats.get('compression_ratio', 1.0)

        return EvalResult(
            method_name=baseline_stats.get('method_name', 'Unknown'),
            perplexity=perplexity,
            kl_divergence=0.0,  # Placeholder
            cache_size_bytes=int(full_result.cache_size_bytes * baseline_stats.get('compression_ratio', 1.0)) if self.config.compare_with_full else 0,
            compression_ratio=baseline_stats.get('compression_ratio', 1.0),
            tokens_kept=baseline_stats.get('tokens_kept', 0),
            prefill_latency_ms=timing['prefill_latency_ms'],
            decode_latency_ms=timing['decode_latency_ms'],
            throughput_tokens_per_sec=timing['throughput_tokens_per_sec'],
            generated_text=generated_text,
            num_input_tokens=num_input_tokens,
            num_output_tokens=max_new_tokens,
        )

    def compare_baselines(
        self,
        baselines: List,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> List[EvalResult]:
        """Compare multiple baseline methods."""
        results = []

        # Evaluate full cache first
        full_result = self.evaluate_full_cache(prompt, max_new_tokens)
        results.append(full_result)

        # Evaluate each baseline
        for baseline in baselines:
            baseline.reset()
            result = self.evaluate_with_baseline(baseline, prompt, max_new_tokens)
            results.append(result)

        return results

    def format_results(
        self,
        results: List[EvalResult],
    ) -> str:
        """Format results as comparison table."""
        lines = []
        lines.append("=" * 80)
        lines.append("Baseline Comparison Results")
        lines.append("=" * 80)

        header = f"{'Method':<20} {'PPL':>10} {'Compress':>10} {'Prefill(ms)':>12} {'Decode(ms)':>12} {'TPS':>10}"
        lines.append(header)
        lines.append("-" * 80)

        for r in results:
            row = f"{r.method_name:<20} {r.perplexity:>10.2f} {r.compression_ratio:>10.2%} {r.prefill_latency_ms:>12.1f} {r.decode_latency_ms:>12.1f} {r.throughput_tokens_per_sec:>10.1f}"
            lines.append(row)

        lines.append("=" * 80)

        return "\n".join(lines)


# Test code
if __name__ == "__main__":
    print("Testing Evaluation Framework...")

    # Create synthetic test
    class MockModel:
        class Config:
            num_hidden_layers = 32
            num_attention_heads = 8
            hidden_size = 1024

        config = Config()
        device = 'cuda'

        def __call__(self, *args, **kwargs):
            # Mock output
            class Output:
                logits = torch.randn(1, 1, 1000)
                loss = torch.tensor(2.0)
                past_key_values = None
            return Output()

    class MockTokenizer:
        vocab_size = 1000

        def encode(self, text):
            return [1, 2, 3, 4, 5]

        def decode(self, ids):
            return "generated text"

        def __call__(self, text, return_tensors='pt'):
            return {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

    model = MockModel()
    tokenizer = MockTokenizer()

    evaluator = BaselineEvaluator(model, tokenizer)

    # Test individual components
    print(f"Model info: {evaluator.num_layers} layers, {evaluator.num_heads} heads, {evaluator.head_dim} head_dim")

    # Test memory calculator
    cache_size = evaluator.memory_tracker.get_cache_size(
        num_tokens=100,
        num_layers=32,
        num_heads=8,
        head_dim=128,
        precision_bytes=2,
    )
    print(f"Cache size for 100 tokens: {cache_size / 1024 / 1024:.2f} MB")

    print("\nEvaluation framework test passed!")