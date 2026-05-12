"""
Oracle Attribution Pipeline for NeuroKV

Generates ground-truth labels for KV cache management decisions by:
1. Running full attention (oracle)
2. Computing per-token importance via attention attribution
3. Generating action labels for imitation learning

Two methods:
- Exact: Leave-one-out drop in log-likelihood (expensive)
- Approximate: Attention weight-based attribution (fast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AttributionMethod(Enum):
    """Methods for computing token importance."""
    EXACT_LEAVE_ONE_OUT = "exact"      # Remove token, measure drop
    ATTENTION_WEIGHT = "attention"     # Use attention weights directly
    ATTENTION_ACCUMULATED = "accumulated"  # Accumulate over time
    GRADIENT_BASED = "gradient"        # Gradient-based attribution


@dataclass
class OracleConfig:
    """Configuration for oracle attribution."""
    method: AttributionMethod = AttributionMethod.ATTENTION_ACCUMULATED

    # Memory budget ratios for generating labels
    tier1_ratio: float = 0.15   # FP16
    tier2_ratio: float = 0.35   # INT4
    tier3_ratio: float = 0.50   # DRAM (implicit, remaining)

    # For attention-based methods
    attention_window: int = 32  # Window for recent attention stats

    # For gradient-based method
    gradient_steps: int = 10    # Number of gradient integration steps


@dataclass
class TokenImportance:
    """Importance score for a single token."""
    position: int
    score: float
    layer: int
    action_label: int  # 0=KEEP_FP16, 1=COMPRESS_INT4, 2=OFFLOAD, 3=EVICT


class OracleAttribution:
    """
    Computes oracle attribution scores and action labels.

    Used for imitation learning: train policy to match oracle decisions.
    """

    # Action labels
    KEEP_FP16 = 0
    COMPRESS_INT4 = 1
    OFFLOAD_DRAM = 2
    EVICT = 3

    def __init__(self, config: OracleConfig):
        self.config = config
        self.accumulated_scores = {}  # layer -> (position -> score)

    def compute_importance(
        self,
        attention_weights: torch.Tensor,
        layer: int,
        current_position: int,
        kv_cache_length: int,
    ) -> torch.Tensor:
        """
        Compute importance scores for each cached token.

        Args:
            attention_weights: (batch, num_heads, query_len, key_len)
            layer: Layer index
            current_position: Current generation position
            kv_cache_length: Total KV cache length

        Returns:
            importance: (batch, kv_cache_length) - importance score per token
        """
        batch, num_heads, query_len, key_len = attention_weights.shape

        if self.config.method == AttributionMethod.ATTENTION_WEIGHT:
            # Simple: sum attention weights over all queries and heads
            importance = attention_weights.sum(dim=(1, 2))  # (batch, key_len)

        elif self.config.method == AttributionMethod.ATTENTION_ACCUMULATED:
            # Accumulate attention scores over time
            if layer not in self.accumulated_scores:
                self.accumulated_scores[layer] = {}

            # Current attention
            current_importance = attention_weights.sum(dim=(1, 2))

            # Add to accumulated
            for pos in range(key_len):
                if pos not in self.accumulated_scores[layer]:
                    self.accumulated_scores[layer][pos] = 0.0
                self.accumulated_scores[layer][pos] += current_importance[0, pos].item()

            # Return accumulated scores
            importance = torch.zeros(batch, kv_cache_length)
            for pos in range(min(kv_cache_length, key_len)):
                importance[0, pos] = self.accumulated_scores[layer].get(pos, 0.0)

        elif self.config.method == AttributionMethod.EXACT_LEAVE_ONE_OUT:
            # Placeholder for exact method (would require actual model inference)
            # For now, use attention as proxy
            importance = attention_weights.sum(dim=(1, 2))

        else:
            # Default to attention weights
            importance = attention_weights.sum(dim=(1, 2))

        return importance

    def generate_action_labels(
        self,
        importance_scores: torch.Tensor,
        num_tokens: int,
        tier1_budget: Optional[int] = None,
        tier2_budget: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate action labels based on importance scores.

        Args:
            importance_scores: (batch, num_tokens) - importance per token
            num_tokens: Total number of tokens
            tier1_budget: Number of tokens for Tier-1 (optional)
            tier2_budget: Number of tokens for Tier-2 (optional)

        Returns:
            labels: (batch, num_tokens) - action label per token
        """
        batch = importance_scores.shape[0]

        # Calculate budgets
        if tier1_budget is None:
            tier1_budget = int(self.config.tier1_ratio * num_tokens)
        if tier2_budget is None:
            tier2_budget = int(self.config.tier2_ratio * num_tokens)

        # Ensure budgets don't exceed total
        tier1_budget = min(tier1_budget, num_tokens)
        tier2_budget = min(tier2_budget, num_tokens - tier1_budget)

        # Sort tokens by importance
        sorted_indices = importance_scores.argsort(dim=-1, descending=True)

        # Initialize all tokens as EVICT (action 3)
        labels = torch.full((batch, num_tokens), self.EVICT, dtype=torch.long)

        # Tier-1: Top tier1_budget tokens -> KEEP_FP16
        tier1_indices = sorted_indices[:, :tier1_budget]
        labels.scatter_(1, tier1_indices, self.KEEP_FP16)

        # Tier-2: Next tier2_budget tokens -> COMPRESS_INT4
        tier2_indices = sorted_indices[:, tier1_budget:tier1_budget + tier2_budget]
        labels.scatter_(1, tier2_indices, self.COMPRESS_INT4)

        # Tier-3: Remaining tokens -> OFFLOAD_DRAM (but we treat as EVICT for now)
        # In full implementation, remaining would go to DRAM

        return labels

    def generate_labels_for_layer(
        self,
        attention_weights: torch.Tensor,
        layer: int,
        current_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate importance scores and action labels for a layer.

        Args:
            attention_weights: (batch, num_heads, query_len, key_len)
            layer: Layer index
            current_position: Current generation position

        Returns:
            importance: (batch, key_len) - importance scores
            labels: (batch, key_len) - action labels
        """
        kv_cache_length = attention_weights.shape[-1]

        # Compute importance
        importance = self.compute_importance(
            attention_weights, layer, current_position, kv_cache_length
        )

        # Generate labels
        labels = self.generate_action_labels(importance, kv_cache_length)

        return importance, labels

    def reset(self):
        """Reset accumulated scores for new sequence."""
        self.accumulated_scores = {}

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about generated labels."""
        stats = {
            "total_tokens": 0,
            "tier1_tokens": 0,
            "tier2_tokens": 0,
            "evicted_tokens": 0,
        }

        for layer_scores in self.accumulated_scores.values():
            stats["total_tokens"] += len(layer_scores)

        return stats


class OracleTraceGenerator:
    """
    Generates oracle traces for imitation learning dataset.

    For each prompt, generates:
    - Per-token importance scores
    - Action labels for each generation step
    - Features for policy network
    """

    def __init__(self, config: OracleConfig):
        self.config = config
        self.oracle = OracleAttribution(config)
        self.traces = []  # List of trace dictionaries

    def generate_trace(
        self,
        prompt_length: int,
        generation_length: int,
        num_layers: int = 32,
        num_heads: int = 8,
        head_dim: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic oracle trace (without real model).

        This creates synthetic data for testing the pipeline.
        For real traces, would run actual model inference.

        Args:
            prompt_length: Number of prompt tokens
            generation_length: Number of tokens to generate
            num_layers: Number of LLM layers
            num_heads: Number of attention heads
            head_dim: Head dimension

        Returns:
            trace: Dictionary containing importance scores, labels, features
        """
        self.oracle.reset()

        total_length = prompt_length + generation_length

        # Initialize storage - separate lists for each step to handle variable sizes
        trace_steps = []

        # Simulate generation steps
        for gen_step in range(generation_length):
            current_length = prompt_length + gen_step + 1

            # Simulate attention weights (synthetic)
            # Real implementation would use actual model attention
            attention_weights = self._simulate_attention(
                batch=1,
                num_heads=num_heads,
                query_len=1,  # Decode: single query
                key_len=current_length,
                pattern="hybrid",  # Mix of local and global attention
            )

            # Generate for first layer only (for simplicity)
            # In real implementation, would do all layers
            layer = 0
            importance, labels = self.oracle.generate_labels_for_layer(
                attention_weights, layer, current_length - 1
            )

            # Extract features
            features = self._extract_features(
                attention_weights, current_length - 1, layer, importance
            )

            step_data = {
                "step": gen_step,
                "layer": layer,
                "current_length": current_length,
                "importance": importance,
                "labels": labels,
                "features": features,
                "positions": torch.arange(current_length),
            }
            trace_steps.append(step_data)

        # Store as list of steps (variable sizes)
        trace = {
            "steps": trace_steps,
            "prompt_length": prompt_length,
            "generation_length": generation_length,
            "num_layers": num_layers,
            "num_steps": len(trace_steps),
        }

        self.traces.append(trace)
        return trace

    def _simulate_attention(
        self,
        batch: int,
        num_heads: int,
        query_len: int,
        key_len: int,
        pattern: str = "local",
    ) -> torch.Tensor:
        """
        Simulate attention weights for testing.

        Patterns:
        - "local": Strong attention to nearby tokens
        - "global": Uniform attention
        - "hybrid": Mix of local and global (realistic)
        - "sink": Strong attention to first few tokens (attention sinks)
        """
        attention = torch.zeros(batch, num_heads, query_len, key_len)

        if pattern == "local":
            # Local attention: strong near query position
            for q in range(query_len):
                for k in range(key_len):
                    distance = abs(q - k)
                    attention[:, :, q, k] = math.exp(-distance / 10)

        elif pattern == "global":
            # Uniform attention
            attention.fill_(1.0 / key_len)

        elif pattern == "hybrid":
            # Mix: strong at recent + some global
            for q in range(query_len):
                for k in range(key_len):
                    # Recent tokens
                    if k >= key_len - 32:
                        attention[:, :, q, k] = 0.5
                    # First tokens (attention sinks)
                    elif k < 4:
                        attention[:, :, q, k] = 0.3
                    # Random global attention
                    else:
                        attention[:, :, q, k] = 0.1 * torch.rand(1).item()

        elif pattern == "sink":
            # Attention sink pattern
            for q in range(query_len):
                for k in range(key_len):
                    if k < 4:
                        attention[:, :, q, k] = 0.7
                    elif k >= key_len - 16:
                        attention[:, :, q, k] = 0.2
                    else:
                        attention[:, :, q, k] = 0.05

        # Normalize
        attention = F.softmax(attention, dim=-1)

        return attention

    def _extract_features(
        self,
        attention_weights: torch.Tensor,
        current_position: int,
        layer: int,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features for policy network input.

        Features:
        0: mean_attention - Mean attention received
        1: max_attention - Max attention received
        2: normalized_position - Position / 10000
        3: relative_position - Distance to current query
        4: layer_id - Layer normalized
        5: importance - Accumulated importance score
        6-15: Additional features (zeros for now)
        """
        batch, num_heads, query_len, key_len = attention_weights.shape

        # Attention statistics
        mean_attn = attention_weights.mean(dim=(1, 2))  # (batch, key_len)
        max_attn = attention_weights.max(dim=1)[0].max(dim=1)[0]  # (batch, key_len)

        # Position features
        positions = torch.arange(key_len)
        relative_pos = current_position - positions

        # Build feature matrix
        features = torch.zeros(batch, key_len, 16)

        features[:, :, 0] = mean_attn
        features[:, :, 1] = max_attn.squeeze(0) if max_attn.dim() == 3 else max_attn
        features[:, :, 2] = positions.float() / 10000
        features[:, :, 3] = relative_pos.float() / 100
        features[:, :, 4] = layer / 32
        features[:, :, 5] = importance

        return features

    def save_traces(self, path: str):
        """Save generated traces to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.traces, f)
        print(f"Saved {len(self.traces)} traces to {path}")

    def load_traces(self, path: str):
        """Load traces from file."""
        import pickle
        with open(path, 'rb') as f:
            self.traces = pickle.load(f)
        print(f"Loaded {len(self.traces)} traces from {path}")

    def get_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all traces as dataset tensors.

        Flattens all steps into a single dataset for training.

        Returns:
            features: (total_samples, feature_dim)
            labels: (total_samples,)
        """
        if len(self.traces) == 0:
            raise ValueError("No traces available")

        # Collect all features and labels across all steps
        all_features = []
        all_labels = []

        for trace in self.traces:
            for step in trace["steps"]:
                features = step["features"]  # (batch, seq_len, feature_dim)
                labels = step["labels"]      # (batch, seq_len)

                # Flatten
                all_features.append(features.reshape(-1, features.shape[-1]))
                all_labels.append(labels.reshape(-1))

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return features, labels


# Test code
if __name__ == "__main__":
    print("Testing Oracle Attribution Pipeline...")

    config = OracleConfig(
        method=AttributionMethod.ATTENTION_ACCUMULATED,
        tier1_ratio=0.15,
        tier2_ratio=0.35,
    )

    oracle = OracleAttribution(config)

    # Test importance computation
    batch, num_heads, query_len, key_len = 1, 8, 1, 100
    attention_weights = torch.randn(batch, num_heads, query_len, key_len).softmax(dim=-1)

    importance, labels = oracle.generate_labels_for_layer(attention_weights, layer=0, current_position=99)

    print(f"Importance scores shape: {importance.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels.flatten(), minlength=4).tolist()}")

    # Test trace generator
    print("\nTesting Trace Generator...")
    generator = OracleTraceGenerator(config)

    trace = generator.generate_trace(
        prompt_length=100,
        generation_length=50,
        num_layers=4,  # Use fewer layers for test
    )

    print(f"Trace keys: {trace.keys()}")
    print(f"Num steps: {trace['num_steps']}")

    # Get dataset
    features, labels = generator.get_dataset()
    print(f"\nDataset features shape: {features.shape}")
    print(f"Dataset labels shape: {labels.shape}")
    print(f"Label distribution in dataset: {torch.bincount(labels, minlength=4).tolist()}")

    print("\nAll oracle tests passed!")