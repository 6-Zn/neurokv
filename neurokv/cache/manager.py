"""
NeuroKV Cache Manager

Manages three-tier hierarchical KV cache:
- Tier-1: FP16 HBM (hot cache)
- Tier-2: INT4 HBM (warm cache, quantized)
- Tier-3: INT4 DRAM (cold cache, offloaded)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class CacheTier(Enum):
    """Cache tier enumeration."""
    TIER_1_FP16_HBM = 0  # Full precision in HBM
    TIER_2_INT4_HBM = 1  # Quantized in HBM
    TIER_3_INT4_DRAM = 2  # Quantized in DRAM (offloaded)
    EVICTED = 3  # Completely removed


class CacheAction(Enum):
    """Actions for cache management."""
    KEEP_FP16 = 0
    COMPRESS_INT4 = 1
    OFFLOAD_DRAM = 2
    EVICT = 3


@dataclass
class CacheTokenInfo:
    """Information about a cached token."""
    position: int
    tier: CacheTier
    last_attention_score: float
    accumulated_attention: float
    staleness: int  # Steps since last strong attention


@dataclass
class KVCacheEntry:
    """A single KV cache entry."""
    key: torch.Tensor  # (num_heads, head_dim)
    value: torch.Tensor  # (num_heads, head_dim)
    position: int
    tier: CacheTier

    # For quantized entries
    quantized_key: Optional[torch.Tensor] = None
    quantized_value: Optional[torch.Tensor] = None
    key_scale: Optional[torch.Tensor] = None
    value_scale: Optional[torch.Tensor] = None
    key_min: Optional[torch.Tensor] = None
    value_min: Optional[torch.Tensor] = None


class Quantizer:
    """
    Group-wise asymmetric quantizer for KV cache.

    Based on KIVI's quantization approach:
    - Keys: per-channel quantization
    - Values: per-token quantization
    """

    def __init__(self, num_bits: int = 4, group_size: int = 128):
        self.num_bits = num_bits
        self.group_size = group_size
        self.max_val = 2 ** num_bits - 1

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor with group-wise asymmetric quantization.

        Args:
            tensor: (batch, num_heads, seq_len, head_dim) or similar

        Returns:
            quantized: Quantized tensor (integers)
            scale: Per-group scale
            min_val: Per-group minimum
        """
        # Reshape for group-wise quantization
        original_shape = tensor.shape

        # Flatten and group
        flat = tensor.reshape(-1)
        num_elements = flat.shape[0]
        num_groups = num_elements // self.group_size

        if num_elements % self.group_size != 0:
            # Pad to full groups
            pad_size = (num_groups + 1) * self.group_size - num_elements
            flat = torch.cat([flat, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)])
            num_groups += 1

        grouped = flat.reshape(num_groups, self.group_size)

        # Compute min/max per group
        min_val = grouped.min(dim=-1)[0]
        max_val = grouped.max(dim=-1)[0]

        # Scale
        scale = (max_val - min_val) / self.max_val
        scale = scale.clamp(min=1e-8)  # Avoid division by zero

        # Quantize
        normalized = (grouped - min_val.unsqueeze(-1)) / scale.unsqueeze(-1)
        quantized = normalized.clamp(0, self.max_val).round()

        return quantized, scale, min_val

    def dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        min_val: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Dequantize back to floating point.

        Args:
            quantized: Quantized integers
            scale: Per-group scale
            min_val: Per-group minimum
            original_shape: Target shape

        Returns:
            tensor: Dequantized tensor
        """
        # Dequantize
        dequantized = quantized * scale.unsqueeze(-1) + min_val.unsqueeze(-1)

        # Reshape back
        flat = dequantized.reshape(-1)
        # Compute total elements from original shape
        num_elements = 1
        for dim in original_shape:
            num_elements *= dim
        flat = flat[:num_elements]

        return flat.reshape(original_shape)


class ThreeTierCacheManager:
    """
    Manages three-tier hierarchical KV cache.

    Tiers:
    - Tier-1 (Hot): FP16 in HBM, ~10-20% of tokens
    - Tier-2 (Warm): INT4 in HBM, ~30-50% of tokens
    - Tier-3 (Cold): INT4 in DRAM, remaining tokens

    Actions:
    - KEEP_FP16: Keep in Tier-1
    - COMPRESS_INT4: Move to Tier-2 (quantize)
    - OFFLOAD_DRAM: Move to Tier-3 (offload)
    - EVICT: Remove from cache entirely
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tier1_budget_ratio: float = 0.15,  # 15% in FP16
        tier2_budget_ratio: float = 0.35,  # 35% in INT4 HBM
        tier3_budget_ratio: float = 0.50,  # 50% in INT4 DRAM
        quantize_bits: int = 4,
        group_size: int = 128,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        self.tier1_budget_ratio = tier1_budget_ratio
        self.tier2_budget_ratio = tier2_budget_ratio
        self.tier3_budget_ratio = tier3_budget_ratio

        self.quantizer = Quantizer(num_bits=quantize_bits, group_size=group_size)

        # Initialize tier storage
        self._init_storage()

        # Statistics tracking
        self.stats = {
            "tier1_tokens": 0,
            "tier2_tokens": 0,
            "tier3_tokens": 0,
            "evicted_tokens": 0,
            "tier_transitions": 0,
        }

    def _init_storage(self):
        """Initialize storage for each tier."""
        # Tier-1: FP16 in HBM
        self.tier1_cache: Dict[int, List[KVCacheEntry]] = {
            layer: [] for layer in range(self.num_layers)
        }

        # Tier-2: INT4 in HBM
        self.tier2_cache: Dict[int, List[KVCacheEntry]] = {
            layer: [] for layer in range(self.num_layers)
        }

        # Tier-3: INT4 in DRAM (CPU memory)
        self.tier3_cache: Dict[int, List[KVCacheEntry]] = {
            layer: [] for layer in range(self.num_layers)
        }

    def add_tokens(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        Add new tokens to cache (initially in Tier-1).

        Args:
            layer: Layer index
            keys: (batch, num_heads, new_tokens, head_dim)
            values: (batch, num_heads, new_tokens, head_dim)
            positions: (batch, new_tokens)
        """
        batch_size = keys.shape[0]
        new_tokens = keys.shape[2]

        for b in range(batch_size):
            for t in range(new_tokens):
                entry = KVCacheEntry(
                    key=keys[b, :, t, :].clone(),
                    value=values[b, :, t, :].clone(),
                    position=positions[b, t].item(),
                    tier=CacheTier.TIER_1_FP16_HBM,
                )
                self.tier1_cache[layer].append(entry)

        self.stats["tier1_tokens"] += new_tokens * batch_size

    def apply_policy(
        self,
        layer: int,
        actions: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        Apply policy actions to move tokens between tiers.

        Args:
            layer: Layer index
            actions: (batch, seq_len) - action for each token
            positions: (batch, seq_len) - positions of tokens
        """
        # Find entries by position and apply actions
        for action_idx in range(4):
            mask = actions == action_idx
            if not mask.any():
                continue

            affected_positions = positions[mask].tolist()
            action = CacheAction(action_idx)

            for pos in affected_positions:
                entry = self._find_entry(layer, pos)
                if entry is None:
                    continue

                self._apply_action(layer, entry, action)

    def _find_entry(self, layer: int, position: int) -> Optional[KVCacheEntry]:
        """Find cache entry by position across all tiers."""
        for tier_cache in [self.tier1_cache, self.tier2_cache, self.tier3_cache]:
            for entry in tier_cache[layer]:
                if entry.position == position:
                    return entry
        return None

    def _apply_action(self, layer: int, entry: KVCacheEntry, action: CacheAction) -> None:
        """Apply action to move entry between tiers."""
        current_tier = entry.tier

        if action == CacheAction.KEEP_FP16:
            # Keep in Tier-1
            if current_tier != CacheTier.TIER_1_FP16_HBM:
                self._promote_to_tier1(layer, entry)

        elif action == CacheAction.COMPRESS_INT4:
            # Move to Tier-2 (quantize)
            if current_tier == CacheTier.TIER_1_FP16_HBM:
                self._compress_to_tier2(layer, entry)
            elif current_tier == CacheTier.TIER_3_INT4_DRAM:
                self._load_from_tier3_to_tier2(layer, entry)

        elif action == CacheAction.OFFLOAD_DRAM:
            # Move to Tier-3 (offload)
            if current_tier in [CacheTier.TIER_1_FP16_HBM, CacheTier.TIER_2_INT4_HBM]:
                self._offload_to_tier3(layer, entry)

        elif action == CacheAction.EVICT:
            # Remove from cache
            self._evict_entry(layer, entry)

        self.stats["tier_transitions"] += 1

    def _promote_to_tier1(self, layer: int, entry: KVCacheEntry) -> None:
        """Promote entry to Tier-1 (full precision)."""
        if entry.tier == CacheTier.TIER_2_INT4_HBM:
            # Dequantize
            entry.key = self.quantizer.dequantize(
                entry.quantized_key, entry.key_scale, entry.key_min,
                (self.num_heads, self.head_dim)
            )
            entry.value = self.quantizer.dequantize(
                entry.quantized_value, entry.value_scale, entry.value_min,
                (self.num_heads, self.head_dim)
            )
            # Clear quantized data
            entry.quantized_key = None
            entry.quantized_value = None

        elif entry.tier == CacheTier.TIER_3_INT4_DRAM:
            # Load from DRAM and dequantize
            entry.quantized_key = entry.quantized_key.to(self.device)
            entry.quantized_value = entry.quantized_value.to(self.device)
            self._promote_to_tier1(layer, entry)  # Recursively dequantize

        # Remove from old tier
        self._remove_from_tier(layer, entry.tier, entry)
        # Add to Tier-1
        entry.tier = CacheTier.TIER_1_FP16_HBM
        self.tier1_cache[layer].append(entry)
        self.stats["tier1_tokens"] += 1

    def _compress_to_tier2(self, layer: int, entry: KVCacheEntry) -> None:
        """Quantize and move to Tier-2."""
        # Quantize
        entry.quantized_key, entry.key_scale, entry.key_min = self.quantizer.quantize(entry.key)
        entry.quantized_value, entry.value_scale, entry.value_min = self.quantizer.quantize(entry.value)

        # Clear FP16 data
        entry.key = None
        entry.value = None

        # Remove from Tier-1
        self._remove_from_tier(layer, CacheTier.TIER_1_FP16_HBM, entry)
        # Add to Tier-2
        entry.tier = CacheTier.TIER_2_INT4_HBM
        self.tier2_cache[layer].append(entry)
        self.stats["tier1_tokens"] -= 1
        self.stats["tier2_tokens"] += 1

    def _offload_to_tier3(self, layer: int, entry: KVCacheEntry) -> None:
        """Offload quantized data to DRAM."""
        if entry.tier == CacheTier.TIER_1_FP16_HBM:
            # First quantize
            self._compress_to_tier2(layer, entry)

        # Move to CPU memory
        entry.quantized_key = entry.quantized_key.to("cpu")
        entry.quantized_value = entry.quantized_value.to("cpu")
        entry.key_scale = entry.key_scale.to("cpu")
        entry.value_scale = entry.value_scale.to("cpu")
        entry.key_min = entry.key_min.to("cpu")
        entry.value_min = entry.value_min.to("cpu")

        # Remove from Tier-2
        self._remove_from_tier(layer, CacheTier.TIER_2_INT4_HBM, entry)
        # Add to Tier-3
        entry.tier = CacheTier.TIER_3_INT4_DRAM
        self.tier3_cache[layer].append(entry)
        self.stats["tier2_tokens"] -= 1
        self.stats["tier3_tokens"] += 1

    def _load_from_tier3_to_tier2(self, layer: int, entry: KVCacheEntry) -> None:
        """Load from Tier-3 back to Tier-2."""
        # Move to GPU
        entry.quantized_key = entry.quantized_key.to(self.device)
        entry.quantized_value = entry.quantized_value.to(self.device)
        entry.key_scale = entry.key_scale.to(self.device)
        entry.value_scale = entry.value_scale.to(self.device)
        entry.key_min = entry.key_min.to(self.device)
        entry.value_min = entry.value_min.to(self.device)

        # Remove from Tier-3
        self._remove_from_tier(layer, CacheTier.TIER_3_INT4_DRAM, entry)
        # Add to Tier-2
        entry.tier = CacheTier.TIER_2_INT4_HBM
        self.tier2_cache[layer].append(entry)
        self.stats["tier3_tokens"] -= 1
        self.stats["tier2_tokens"] += 1

    def _evict_entry(self, layer: int, entry: KVCacheEntry) -> None:
        """Remove entry from cache."""
        self._remove_from_tier(layer, entry.tier, entry)

        # Update stats
        if entry.tier == CacheTier.TIER_1_FP16_HBM:
            self.stats["tier1_tokens"] -= 1
        elif entry.tier == CacheTier.TIER_2_INT4_HBM:
            self.stats["tier2_tokens"] -= 1
        elif entry.tier == CacheTier.TIER_3_INT4_DRAM:
            self.stats["tier3_tokens"] -= 1

        self.stats["evicted_tokens"] += 1

    def _remove_from_tier(self, layer: int, tier: CacheTier, entry: KVCacheEntry) -> None:
        """Remove entry from specific tier."""
        tier_cache = self._get_tier_cache(tier)
        tier_cache[layer] = [e for e in tier_cache[layer] if e.position != entry.position]

    def _get_tier_cache(self, tier: CacheTier) -> Dict[int, List[KVCacheEntry]]:
        """Get cache dict for tier."""
        if tier == CacheTier.TIER_1_FP16_HBM:
            return self.tier1_cache
        elif tier == CacheTier.TIER_2_INT4_HBM:
            return self.tier2_cache
        elif tier == CacheTier.TIER_3_INT4_DRAM:
            return self.tier3_cache
        return {}

    def get_attention_input(
        self,
        layer: int,
        query_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather KV cache for attention computation.

        Args:
            layer: Layer index
            query_positions: Positions being queried

        Returns:
            keys: (batch, num_heads, seq_len, head_dim)
            values: (batch, num_heads, seq_len, head_dim)
        """
        # Collect all entries
        all_entries = []
        all_entries.extend(self.tier1_cache[layer])
        all_entries.extend(self.tier2_cache[layer])
        all_entries.extend(self.tier3_cache[layer])

        # Sort by position
        all_entries.sort(key=lambda e: e.position)

        # Gather keys and values
        keys = []
        values = []

        for entry in all_entries:
            if entry.tier == CacheTier.TIER_1_FP16_HBM:
                keys.append(entry.key)
                values.append(entry.value)
            elif entry.tier == CacheTier.TIER_2_INT4_HBM:
                # Dequantize on the fly
                key = self.quantizer.dequantize(
                    entry.quantized_key, entry.key_scale, entry.key_min,
                    (self.num_heads, self.head_dim)
                )
                value = self.quantizer.dequantize(
                    entry.quantized_value, entry.value_scale, entry.value_min,
                    (self.num_heads, self.head_dim)
                )
                keys.append(key)
                values.append(value)
            elif entry.tier == CacheTier.TIER_3_INT4_DRAM:
                # Load from DRAM and dequantize
                key = self.quantizer.dequantize(
                    entry.quantized_key.to(self.device),
                    entry.key_scale.to(self.device),
                    entry.key_min.to(self.device),
                    (self.num_heads, self.head_dim)
                )
                value = self.quantizer.dequantize(
                    entry.quantized_value.to(self.device),
                    entry.value_scale.to(self.device),
                    entry.value_min.to(self.device),
                    (self.num_heads, self.head_dim)
                )
                keys.append(key)
                values.append(value)

        if len(keys) == 0:
            return None, None

        # Stack into tensors
        keys = torch.stack(keys, dim=0)  # (seq_len, num_heads, head_dim)
        values = torch.stack(values, dim=0)

        # Transpose to (1, num_heads, seq_len, head_dim) for batch=1
        keys = keys.unsqueeze(0).transpose(1, 2)
        values = values.unsqueeze(0).transpose(1, 2)

        return keys, values

    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage for each tier in bytes."""
        tier1_size = self.stats["tier1_tokens"] * self.num_layers * self.num_heads * self.head_dim * 2 * 2  # FP16 KV
        tier2_size = self.stats["tier2_tokens"] * self.num_layers * self.num_heads * self.head_dim * 2 * (self.quantizer.num_bits / 8)  # INT4 KV
        tier3_size = self.stats["tier3_tokens"] * self.num_layers * self.num_heads * self.head_dim * 2 * (self.quantizer.num_bits / 8)  # INT4 KV in DRAM

        return {
            "tier1_hbm": tier1_size,
            "tier2_hbm": tier2_size,
            "tier3_dram": tier3_size,
            "total": tier1_size + tier2_size + tier3_size,
        }

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return self.stats.copy()


# Two-tier simplified version for initial experiments
class TwoTierCacheManager:
    """
    Simplified two-tier cache manager:
    - Tier-1: FP16 HBM
    - Tier-2: Evicted (removed)

    For initial experiments before full three-tier implementation.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        budget_ratio: float = 0.20,  # Keep 20% in cache
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.budget_ratio = budget_ratio

        # Cache storage
        self.cache: Dict[int, List[KVCacheEntry]] = {
            layer: [] for layer in range(num_layers)
        }

        # Statistics
        self.stats = {
            "cached_tokens": 0,
            "evicted_tokens": 0,
        }

    def add_tokens(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Add tokens to cache."""
        batch_size = keys.shape[0]
        new_tokens = keys.shape[2]

        for b in range(batch_size):
            for t in range(new_tokens):
                entry = KVCacheEntry(
                    key=keys[b, :, t, :].clone(),
                    value=values[b, :, t, :].clone(),
                    position=positions[b, t].item(),
                    tier=CacheTier.TIER_1_FP16_HBM,
                )
                self.cache[layer].append(entry)

        self.stats["cached_tokens"] += new_tokens * batch_size

    def apply_eviction(self, layer: int, keep_mask: torch.Tensor) -> None:
        """Evict tokens not in keep_mask."""
        positions_to_keep = keep_mask.nonzero().flatten().tolist()

        # Filter cache
        before_count = len(self.cache[layer])
        self.cache[layer] = [e for e in self.cache[layer] if e.position in positions_to_keep]
        after_count = len(self.cache[layer])

        self.stats["cached_tokens"] = after_count
        self.stats["evicted_tokens"] += before_count - after_count

    def get_attention_input(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get keys and values for attention."""
        entries = sorted(self.cache[layer], key=lambda e: e.position)

        if len(entries) == 0:
            return None, None

        keys = torch.stack([e.key for e in entries], dim=0)
        values = torch.stack([e.value for e in entries], dim=0)

        keys = keys.unsqueeze(0).transpose(1, 2)
        values = values.unsqueeze(0).transpose(1, 2)

        return keys, values


# Test code
if __name__ == "__main__":
    # Test quantizer
    quantizer = Quantizer(num_bits=4, group_size=128)
    test_tensor = torch.randn(32, 8, 100, 128)  # Simulated KV cache
    quantized, scale, min_val = quantizer.quantize(test_tensor)
    dequantized = quantizer.dequantize(quantized, scale, min_val, test_tensor.shape)

    error = (test_tensor - dequantized).abs().mean()
    print(f"Quantization error (4-bit): {error:.6f}")

    # Test two-tier cache manager
    cache_mgr = TwoTierCacheManager(
        num_layers=32,
        num_heads=8,
        head_dim=128,
    )
    print(f"Two-tier cache initialized")

    # Test three-tier cache manager
    cache_mgr_full = ThreeTierCacheManager(
        num_layers=32,
        num_heads=8,
        head_dim=128,
    )
    print(f"Three-tier cache initialized")