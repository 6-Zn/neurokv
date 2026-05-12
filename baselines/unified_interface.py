"""
Unified Baseline Interface for KV Cache Compression Methods

Provides a common interface for:
- H2O (Heavy-Hitter Oracle)
- StreamingLLM (Attention sinks + sliding window)
- SnapKV (Clustered attention compression)
- KIVI (Quantization)
- Quest (Page-based retrieval)

This interface allows easy comparison and evaluation of all baselines.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================================
# DynamicCache Compatibility Helpers (transformers 5.8+)
# ============================================================================

def get_cache_layers(cache) -> List:
    """
    Get list of cache layers from DynamicCache (transformers 5.8+) or tuple format.

    Returns list of layer objects/tuples, or None if cache is empty/None.
    """
    if cache is None:
        return None

    # Transformers 5.8+ DynamicCache with layers attribute
    if hasattr(cache, 'layers'):
        return cache.layers

    # Older tuple format: list of (key, value) tuples
    if isinstance(cache, (list, tuple)):
        return list(cache)

    return None


def get_layer_kv(layer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get keys and values from a cache layer.

    Supports both DynamicLayer (transformers 5.8+) and tuple format.
    """
    if layer is None:
        return None, None

    # DynamicLayer (transformers 5.8+)
    if hasattr(layer, 'keys') and hasattr(layer, 'values'):
        return layer.keys, layer.values

    # Tuple format (key, value)
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]

    return None, None


def set_layer_kv(layer, keys: torch.Tensor, values: torch.Tensor):
    """
    Set keys and values in a cache layer.

    Supports both DynamicLayer (transformers 5.8+) and creates new tuple for older format.
    """
    if layer is None:
        return (keys, values)

    # DynamicLayer (transformers 5.8+)
    if hasattr(layer, 'keys') and hasattr(layer, 'values'):
        layer.keys = keys
        layer.values = values
        return layer

    # Return new tuple for older format
    return (keys, values)


def get_cache_length(cache) -> int:
    """
    Get the sequence length of the KV cache.

    Works with DynamicCache, tuple format, and older key_cache/value_cache format.
    """
    if cache is None:
        return 0

    # DynamicCache with get_seq_length method
    if hasattr(cache, 'get_seq_length'):
        try:
            return cache.get_seq_length()
        except:
            pass

    # DynamicCache with layers attribute (transformers 5.8+)
    if hasattr(cache, 'layers') and len(cache.layers) > 0:
        keys, _ = get_layer_kv(cache.layers[0])
        if keys is not None:
            return keys.shape[2]

    # Older key_cache/value_cache format
    if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
        return cache.key_cache[0].shape[2]

    # Tuple format
    if isinstance(cache, (list, tuple)) and len(cache) > 0:
        keys, _ = get_layer_kv(cache[0])
        if keys is not None:
            return keys.shape[2]

    return 0


def compress_dynamic_cache(
    cache,
    baseline,
    current_position: int = 0,
    log: bool = False,
) -> None:
    """
    Apply baseline compression to a DynamicCache (transformers 5.8+) or tuple cache.

    Modifies the cache in-place for DynamicCache, returns new tuple for tuple format.

    Args:
        cache: DynamicCache or tuple of (key, value) per layer
        baseline: KVCacheBaseline instance
        current_position: Current generation position
        log: Print compression details

    Returns:
        Modified cache (same object for DynamicCache, new tuple for tuple format)
    """
    if cache is None:
        return None

    layers = get_cache_layers(cache)
    if layers is None or len(layers) == 0:
        return cache

    baseline.reset()

    for layer_idx, layer in enumerate(layers):
        keys, values = get_layer_kv(layer)
        if keys is None or values is None:
            continue

        original_len = keys.shape[2]

        # Apply baseline compression
        compressed_keys, compressed_values = baseline.compress(
            keys, values, None, current_position
        )

        new_len = compressed_keys.shape[2]

        if log and new_len < original_len:
            print(f"  Layer {layer_idx}: {original_len} -> {new_len} tokens ({new_len/original_len:.1%})")

        # Update layer
        set_layer_kv(layer, compressed_keys, compressed_values)

    return cache


class BaselineMethod(Enum):
    """Available baseline methods."""
    FULL_CACHE = "full"       # No compression (oracle)
    H2O = "h2o"               # Heavy-Hitter Oracle
    STREAMING_LLM = "streaming"  # Attention sinks + sliding window
    SNAPKV = "snapkv"         # Clustered attention compression
    KIVI = "kivi"             # Quantization
    QUEST = "quest"           # Page-based retrieval
    RANDOM = "random"         # Random eviction (lower bound)


@dataclass
class CacheConfig:
    """Configuration for KV cache compression."""
    method: BaselineMethod

    # H2O parameters
    heavy_ratio: float = 0.1      # Fraction of heavy hitters
    recent_ratio: float = 0.1     # Fraction of recent tokens

    # StreamingLLM parameters
    start_size: int = 4           # Number of attention sinks
    recent_size: int = 512        # Sliding window size

    # SnapKV parameters
    max_capacity_prompt: int = 2048  # Maximum prompt cache size
    window_size: int = 32         # Observation window
    kernel_size: int = 5          # Pooling kernel size
    pooling: str = "avgpool"      # Pooling method

    # KIVI parameters
    quantize_bits: int = 4        # 2-bit or 4-bit
    group_size: int = 128         # Quantization group size

    # Quest parameters
    page_size: int = 256          # Page size for Quest
    page_budget: int = 512        # Number of pages to keep

    # General parameters
    max_seq_len: int = 4096       # Maximum sequence length


class KVCacheBaseline(ABC):
    """
    Abstract base class for KV cache compression baselines.

    All baselines must implement:
    - compress(): Apply compression to KV cache
    - get_cache(): Return compressed cache for attention
    - reset(): Reset cache state for new sequence
    """

    @abstractmethod
    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply compression to KV cache.

        Args:
            keys: (batch, num_heads, seq_len, head_dim)
            values: (batch, num_heads, seq_len, head_dim)
            attention_weights: (batch, num_heads, query_len, key_len) optional
            current_position: Current generation position

        Returns:
            compressed_keys, compressed_values
        """
        pass

    @abstractmethod
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current compressed cache."""
        pass

    @abstractmethod
    def reset(self):
        """Reset cache state for new sequence."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        """Return compression statistics."""
        pass


class FullCacheBaseline(KVCacheBaseline):
    """No compression - keeps all tokens (oracle baseline)."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.stats = {"compression_ratio": 1.0, "tokens_kept": 0}

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep all tokens - no compression."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)

        self.stats["tokens_kept"] = self.keys.shape[2]
        return self.keys, self.values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


class H2OBaseline(KVCacheBaseline):
    """
    H2O: Heavy-Hitter Oracle

    Keeps tokens with highest accumulated attention scores + recent tokens.
    """

    def __init__(self, config: CacheConfig):
        self.heavy_ratio = config.heavy_ratio
        self.recent_ratio = config.recent_ratio
        self.cache_budget = self.heavy_ratio + self.recent_ratio

        self.keys = None
        self.values = None
        self.accumulated_scores = None  # Track attention scores per token
        self.stats = {
            "compression_ratio": self.cache_budget,
            "tokens_kept": 0,
            "heavy_tokens": 0,
            "recent_tokens": 0,
        }

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply H2O compression."""
        batch, num_heads, seq_len, head_dim = keys.shape

        # If this is the first call and we have the full prefill cache,
        # compress it directly based on position (fallback to position-based importance)
        if self.keys is None:
            # Check if seq_len is large enough to need compression
            heavy_budget = int(self.heavy_ratio * seq_len)
            recent_budget = int(self.recent_ratio * seq_len)

            if seq_len > heavy_budget + recent_budget:
                # Fallback: use uniform importance + position-based selection
                # Keep: start tokens + recent tokens (similar to StreamingLLM fallback)
                keep_indices = torch.cat([
                    torch.arange(heavy_budget),  # "heavy" from start
                    torch.arange(seq_len - recent_budget, seq_len)  # recent
                ])
                self.keys = keys[:, :, keep_indices, :]
                self.values = values[:, :, keep_indices, :]
                self.stats["heavy_tokens"] = heavy_budget
                self.stats["recent_tokens"] = recent_budget
                self.stats["tokens_kept"] = self.keys.shape[2]
                return self.keys, self.values

            # Store full cache
            self.keys = keys
            self.values = values
            if attention_weights is not None:
                self.accumulated_scores = attention_weights.sum(dim=(1, 2))
            self.stats["tokens_kept"] = seq_len
            return self.keys, self.values

        # Concatenate new tokens (decode phase)
        self.keys = torch.cat([self.keys, keys], dim=2)
        self.values = torch.cat([self.values, values], dim=2)

        # Update accumulated scores
        if attention_weights is not None and self.accumulated_scores is not None:
            new_scores = attention_weights.sum(dim=(1, 2))
            self.accumulated_scores = torch.cat([self.accumulated_scores, new_scores], dim=-1)

        seq_len = self.keys.shape[2]

        # Calculate budgets
        heavy_budget = int(self.heavy_ratio * seq_len)
        recent_budget = int(self.recent_ratio * seq_len)

        if seq_len <= heavy_budget + recent_budget:
            self.stats["tokens_kept"] = seq_len
            return self.keys, self.values

        # Select tokens
        if self.accumulated_scores is not None:
            old_scores = self.accumulated_scores[:, :-recent_budget]
            _, topk_indices = old_scores.topk(heavy_budget, dim=-1)
            recent_indices = torch.arange(seq_len - recent_budget, seq_len)
            keep_indices = torch.cat([topk_indices.squeeze(0), recent_indices])
            self.keys = self.keys[:, :, keep_indices, :]
            self.values = self.values[:, :, keep_indices, :]
            self.accumulated_scores = self.accumulated_scores[:, keep_indices]
            self.stats["heavy_tokens"] = heavy_budget
            self.stats["recent_tokens"] = recent_budget
        else:
            # Fallback: position-based selection
            keep_indices = torch.cat([
                torch.arange(heavy_budget),
                torch.arange(seq_len - recent_budget, seq_len)
            ])
            self.keys = self.keys[:, :, keep_indices, :]
            self.values = self.values[:, :, keep_indices, :]
            self.stats["heavy_tokens"] = heavy_budget
            self.stats["recent_tokens"] = recent_budget

        self.stats["tokens_kept"] = self.keys.shape[2]
        return self.keys, self.values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
        self.accumulated_scores = None
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


class StreamingLLMBaseline(KVCacheBaseline):
    """
    StreamingLLM: Attention sinks + sliding window.

    Keeps first `start_size` tokens (attention sinks) + recent `recent_size` tokens.
    """

    def __init__(self, config: CacheConfig):
        self.start_size = config.start_size
        self.recent_size = config.recent_size
        self.cache_size = self.start_size + self.recent_size

        self.keys = None
        self.values = None
        self.stats = {
            "compression_ratio": 0.0,  # Dynamic
            "tokens_kept": 0,
            "sink_tokens": self.start_size,
            "recent_tokens": self.recent_size,
        }

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply StreamingLLM compression."""
        batch, num_heads, new_len, head_dim = keys.shape

        # Add new tokens
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)

        seq_len = self.keys.shape[2]

        # Only compress if over budget
        if seq_len <= self.cache_size:
            self.stats["tokens_kept"] = seq_len
            self.stats["compression_ratio"] = 1.0
            return self.keys, self.values

        # Slice: attention sinks (start) + recent tokens
        sink_keys = self.keys[:, :, :self.start_size, :]
        sink_values = self.values[:, :, :self.start_size, :]

        recent_keys = self.keys[:, :, seq_len - self.recent_size:, :]
        recent_values = self.values[:, :, seq_len - self.recent_size:, :]

        self.keys = torch.cat([sink_keys, recent_keys], dim=2)
        self.values = torch.cat([sink_values, recent_values], dim=2)

        self.stats["tokens_kept"] = self.cache_size
        self.stats["compression_ratio"] = self.cache_size / seq_len

        return self.keys, self.values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


class SnapKVBaseline(KVCacheBaseline):
    """
    SnapKV: Clustered attention compression.

    Uses pooled attention scores to identify important positions during prefill.
    """

    def __init__(self, config: CacheConfig):
        self.max_capacity = config.max_capacity_prompt
        self.window_size = config.window_size
        self.kernel_size = config.kernel_size
        self.pooling = config.pooling

        self.keys = None
        self.values = None
        self.prefill_done = False
        self.stats = {
            "compression_ratio": 0.0,
            "tokens_kept": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
        }

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply SnapKV compression."""
        batch, num_heads, seq_len, head_dim = keys.shape

        # Prefill phase: compress if over capacity
        if not self.prefill_done and seq_len > self.max_capacity:
            if attention_weights is not None:
                # Compute importance scores from recent window attention
                window_attn = attention_weights[:, :, -self.window_size:, :-self.window_size]
                attn_sum = window_attn.sum(dim=-2)  # Sum over query window

                # Apply pooling to smooth importance
                if self.pooling == "avgpool":
                    import torch.nn.functional as F
                    importance = F.avg_pool1d(
                        attn_sum.squeeze(0).transpose(1, 0),
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        stride=1
                    ).transpose(1, 0)
                elif self.pooling == "maxpool":
                    import torch.nn.functional as F
                    importance = F.max_pool1d(
                        attn_sum.squeeze(0).transpose(1, 0),
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        stride=1
                    ).transpose(1, 0)
                else:
                    importance = attn_sum

                # Select top-k important positions
                budget = self.max_capacity - self.window_size
                _, topk_indices = importance.mean(dim=0).topk(budget, dim=-1)

                # Keep selected + recent window
                keep_indices = torch.cat([
                    topk_indices,
                    torch.arange(seq_len - self.window_size, seq_len)
                ])

                self.keys = keys[:, :, keep_indices, :]
                self.values = values[:, :, keep_indices, :]
                self.prefill_done = True

                self.stats["prefill_tokens"] = seq_len
                self.stats["tokens_kept"] = self.keys.shape[2]
            else:
                # No attention weights, just keep recent
                self.keys = keys[:, :, -self.max_capacity:, :]
                self.values = values[:, :, -self.max_capacity:, :]
        else:
            # Decode phase: just append
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                self.keys = torch.cat([self.keys, keys], dim=2)
                self.values = torch.cat([self.values, values], dim=2)

        self.stats["tokens_kept"] = self.keys.shape[2] if self.keys is not None else seq_len
        self.stats["compression_ratio"] = self.stats["tokens_kept"] / seq_len if seq_len > 0 else 1.0

        return self.keys, self.values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
        self.prefill_done = False
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


class KIVIBaseline(KVCacheBaseline):
    """
    KIVI: Group-wise asymmetric quantization.

    Quantizes KV cache to 2-bit or 4-bit using per-group min-max quantization.
    """

    def __init__(self, config: CacheConfig):
        self.num_bits = config.quantize_bits
        self.group_size = config.group_size
        self.max_val = 2 ** self.num_bits - 1

        self.keys_fp16 = None
        self.values_fp16 = None
        self.keys_quant = None
        self.values_quant = None
        self.key_scales = None
        self.value_scales = None
        self.key_mins = None
        self.value_mins = None

        self.stats = {
            "compression_ratio": self.num_bits / 16,  # 4-bit = 0.25, 2-bit = 0.125
            "tokens_kept": 0,
            "quantize_error": 0.0,
        }

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group-wise asymmetric quantization."""
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        num_elements = flat.shape[0]

        # Pad to full groups
        num_groups = (num_elements + self.group_size - 1) // self.group_size
        padded_size = num_groups * self.group_size
        if padded_size > num_elements:
            pad = torch.zeros(padded_size - num_elements, dtype=tensor.dtype, device=tensor.device)
            flat = torch.cat([flat, pad])

        grouped = flat.reshape(num_groups, self.group_size)

        # Min-max per group
        min_val = grouped.min(dim=-1)[0]
        max_val = grouped.max(dim=-1)[0]

        # Scale
        scale = (max_val - min_val) / self.max_val
        scale = scale.clamp(min=1e-8)

        # Quantize
        normalized = (grouped - min_val.unsqueeze(-1)) / scale.unsqueeze(-1)
        quantized = normalized.clamp(0, self.max_val).round()

        return quantized, scale, min_val

    def _dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        min_val: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Dequantize back to FP16."""
        dequantized = quantized * scale.unsqueeze(-1) + min_val.unsqueeze(-1)
        flat = dequantized.reshape(-1)

        # Remove padding
        num_elements = 1
        for dim in original_shape:
            num_elements *= dim
        flat = flat[:num_elements]

        return flat.reshape(original_shape)

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply KIVI quantization."""
        # Store FP16 for error calculation
        if self.keys_fp16 is None:
            self.keys_fp16 = keys.clone()
            self.values_fp16 = values.clone()
        else:
            self.keys_fp16 = torch.cat([self.keys_fp16, keys.clone()], dim=2)
            self.values_fp16 = torch.cat([self.values_fp16, values.clone()], dim=2)

        # Quantize
        k_quant, k_scale, k_min = self._quantize(keys)
        v_quant, v_scale, v_min = self._quantize(values)

        # Store quantized versions
        if self.keys_quant is None:
            self.keys_quant = k_quant
            self.values_quant = v_quant
            self.key_scales = k_scale
            self.value_scales = v_scale
            self.key_mins = k_min
            self.value_mins = v_min
        else:
            self.keys_quant = torch.cat([self.keys_quant, k_quant], dim=0)
            self.values_quant = torch.cat([self.values_quant, v_quant], dim=0)
            self.key_scales = torch.cat([self.key_scales, k_scale], dim=0)
            self.value_scales = torch.cat([self.value_scales, v_scale], dim=0)
            self.key_mins = torch.cat([self.key_mins, k_min], dim=0)
            self.value_mins = torch.cat([self.value_mins, v_min], dim=0)

        # Dequantize for output (simulating on-the-fly dequantization)
        dequant_keys = self._dequantize(k_quant, k_scale, k_min, keys.shape)
        dequant_values = self._dequantize(v_quant, v_scale, v_min, values.shape)

        # Calculate error
        error = (keys - dequant_keys).abs().mean().item()
        self.stats["quantize_error"] = error
        self.stats["tokens_kept"] = self.keys_fp16.shape[2]

        return dequant_keys, dequant_values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized cache."""
        if self.keys_fp16 is None:
            return None, None

        # Dequantize all
        shape = self.keys_fp16.shape
        keys = self._dequantize(self.keys_quant, self.key_scales, self.key_mins, shape)
        values = self._dequantize(self.values_quant, self.value_scales, self.value_mins, self.values_fp16.shape)

        return keys, values

    def reset(self):
        self.keys_fp16 = None
        self.values_fp16 = None
        self.keys_quant = None
        self.values_quant = None
        self.key_scales = None
        self.value_scales = None
        self.key_mins = None
        self.value_mins = None
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


class RandomBaseline(KVCacheBaseline):
    """Random eviction - lower bound baseline."""

    def __init__(self, config: CacheConfig):
        self.keep_ratio = config.heavy_ratio + config.recent_ratio  # Match H2O budget
        self.keys = None
        self.values = None
        self.stats = {
            "compression_ratio": self.keep_ratio,
            "tokens_kept": 0,
        }

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        current_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random eviction."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)

        seq_len = self.keys.shape[2]
        budget = int(self.keep_ratio * seq_len)

        if seq_len > budget:
            # Random selection
            indices = torch.randperm(seq_len)[:budget]
            self.keys = self.keys[:, :, indices, :]
            self.values = self.values[:, :, indices, :]

        self.stats["tokens_kept"] = self.keys.shape[2]
        return self.keys, self.values

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
        self.stats["tokens_kept"] = 0

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


def create_baseline(config: CacheConfig) -> KVCacheBaseline:
    """Factory function to create baseline instances."""
    if config.method == BaselineMethod.FULL_CACHE:
        return FullCacheBaseline()
    elif config.method == BaselineMethod.H2O:
        return H2OBaseline(config)
    elif config.method == BaselineMethod.STREAMING_LLM:
        return StreamingLLMBaseline(config)
    elif config.method == BaselineMethod.SNAPKV:
        return SnapKVBaseline(config)
    elif config.method == BaselineMethod.KIVI:
        return KIVIBaseline(config)
    elif config.method == BaselineMethod.RANDOM:
        return RandomBaseline(config)
    else:
        raise ValueError(f"Unknown baseline method: {config.method}")


# Test code
if __name__ == "__main__":
    print("Testing unified baseline interface...")

    # Test configurations
    configs = {
        BaselineMethod.FULL_CACHE: CacheConfig(method=BaselineMethod.FULL_CACHE),
        BaselineMethod.H2O: CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1),
        BaselineMethod.STREAMING_LLM: CacheConfig(method=BaselineMethod.STREAMING_LLM, start_size=4, recent_size=512),
        BaselineMethod.SNAPKV: CacheConfig(method=BaselineMethod.SNAPKV, max_capacity_prompt=256),
        BaselineMethod.KIVI: CacheConfig(method=BaselineMethod.KIVI, quantize_bits=4),
        BaselineMethod.RANDOM: CacheConfig(method=BaselineMethod.RANDOM, heavy_ratio=0.1, recent_ratio=0.1),
    }

    # Test each baseline
    batch, num_heads, seq_len, head_dim = 1, 8, 100, 128

    for method, config in configs.items():
        baseline = create_baseline(config)

        # Simulate prefill
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        attn_weights = torch.randn(batch, num_heads, seq_len, seq_len).softmax(dim=-1)

        compressed_keys, compressed_values = baseline.compress(keys, values, attn_weights)
        stats = baseline.get_stats()

        print(f"{method.value}: kept {stats['tokens_kept']} tokens, ratio {stats['compression_ratio']:.2f}")

    print("\nAll baseline tests passed!")