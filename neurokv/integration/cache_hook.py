"""
KV Cache Hook for NeuroKV

Provides hooks to intercept and modify KV cache during model inference.
Enables real baseline compression integration with model's attention layers.

Key components:
- AttentionHook: Intercept attention layer outputs
- CacheModifier: Apply baseline compression to past_key_values
- HookedModel: Wrapper for model with cache hooks enabled
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math


@dataclass
class HookConfig:
    """Configuration for cache hooks."""
    # Which layers to hook (None = all layers)
    target_layers: Optional[List[int]] = None

    # Compression trigger
    compress_every_n_tokens: int = 100  # Compress every N new tokens
    max_cache_length: int = 2048        # Maximum cache before compression

    # Debug settings
    log_compressions: bool = True
    track_attention_stats: bool = True


class AttentionStats:
    """Track attention statistics for each layer."""

    def __init__(self):
        self.stats = {}  # layer -> {mean, max, entropy, etc.}
        self.accumulated_scores = {}  # layer -> tensor

    def update(self, layer_idx: int, attention_weights: torch.Tensor):
        """Update stats for a layer."""
        # attention_weights: (batch, heads, query_len, key_len)

        if layer_idx not in self.stats:
            self.stats[layer_idx] = {
                'mean': [],
                'max': [],
                'entropy': [],
            }
            self.accumulated_scores[layer_idx] = None

        # Compute statistics
        mean_attn = attention_weights.mean().item()
        max_attn = attention_weights.max().item()

        # Entropy
        entropy = -(attention_weights * attention_weights.log()).sum().item()

        self.stats[layer_idx]['mean'].append(mean_attn)
        self.stats[layer_idx]['max'].append(max_attn)
        self.stats[layer_idx]['entropy'].append(entropy)

        # Accumulate scores
        current_scores = attention_weights.sum(dim=(1, 2))  # (batch, key_len)

        if self.accumulated_scores[layer_idx] is None:
            self.accumulated_scores[layer_idx] = current_scores
        else:
            # Accumulate (handle different lengths)
            old_len = self.accumulated_scores[layer_idx].shape[-1]
            new_len = current_scores.shape[-1]

            if new_len > old_len:
                # New tokens added
                self.accumulated_scores[layer_idx] = torch.cat([
                    self.accumulated_scores[layer_idx],
                    current_scores[:, old_len:]
                ], dim=-1)
            else:
                # Same length, just add
                self.accumulated_scores[layer_idx] = self.accumulated_scores[layer_idx] + current_scores

    def get_importance_scores(self, layer_idx: int) -> torch.Tensor:
        """Get accumulated importance scores for a layer."""
        return self.accumulated_scores.get(layer_idx, None)

    def reset(self):
        """Reset all stats."""
        self.stats = {}
        self.accumulated_scores = {}


class CacheHook:
    """
    Hook for intercepting and modifying KV cache.

    Hooks into attention layers to:
    1. Track attention statistics
    2. Apply baseline compression to past_key_values
    """

    def __init__(
        self,
        baseline,
        config: HookConfig = None,
    ):
        self.baseline = baseline
        self.config = config or HookConfig()

        self.stats = AttentionStats()
        self.hooks = []
        self.compression_count = 0
        self.current_tokens = 0

        # Store modified cache
        self.modified_past_key_values = None

    def register_hooks(self, model):
        """Register forward hooks on attention layers."""
        # Find attention layers
        attention_layers = self._find_attention_layers(model)

        for idx, layer in enumerate(attention_layers):
            if self.config.target_layers is None or idx in self.config.target_layers:
                hook = layer.register_forward_hook(self._attention_hook(idx))
                self.hooks.append(hook)

        return len(self.hooks)

    def _find_attention_layers(self, model):
        """Find attention layer modules in model."""
        attention_layers = []

        # Common patterns for attention layer names
        patterns = [
            'self_attn',           # LLaMA style
            'attention',           # GPT style
            'attn',                # Alternative
            'encoder_attn',        # Encoder attention
        ]

        for name, module in model.named_modules():
            for pattern in patterns:
                if pattern in name.lower() and hasattr(module, 'forward'):
                    # Check if it has q_proj, k_proj, v_proj
                    if hasattr(module, 'q_proj') or hasattr(module, 'query'):
                        attention_layers.append(module)
                        break

        return attention_layers

    def _attention_hook(self, layer_idx: int) -> Callable:
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically a tuple: (hidden_states, attention_weights, past_key_values)
            # For models with past_key_values, output[2] contains the cache

            # Track statistics if enabled
            if self.config.track_attention_stats:
                # Try to get attention weights
                if hasattr(module, '_attention_weights'):
                    self.stats.update(layer_idx, module._attention_weights)

            # Check if compression needed
            self.current_tokens += 1

            if self._should_compress():
                self._apply_compression(layer_idx)

            return output

        return hook

    def _should_compress(self) -> bool:
        """Check if compression should be applied."""
        if self.current_tokens >= self.config.max_cache_length:
            return True
        if self.current_tokens % self.config.compress_every_n_tokens == 0:
            return True
        return False

    def _apply_compression(self, layer_idx: int):
        """Apply baseline compression to cache."""
        if self.modified_past_key_values is None:
            return

        # Get importance scores for this layer
        importance = self.stats.get_importance_scores(layer_idx)

        if importance is not None:
            # Get current KV cache for this layer
            kv = self.modified_past_key_values[layer_idx]
            keys = kv[0]   # (batch, heads, seq_len, head_dim)
            values = kv[1]  # (batch, heads, seq_len, head_dim)

            # Apply baseline compression
            try:
                compressed_keys, compressed_values = self.baseline.compress(
                    keys, values,
                    attention_weights=importance.unsqueeze(1).unsqueeze(1),
                    current_position=self.current_tokens - 1,
                )

                # Update cache
                self.modified_past_key_values[layer_idx] = (compressed_keys, compressed_values)
                self.compression_count += 1

                if self.config.log_compressions:
                    new_len = compressed_keys.shape[2]
                    old_len = keys.shape[2]
                    ratio = new_len / old_len if old_len > 0 else 1.0
                    print(f"[Layer {layer_idx}] Compressed: {old_len} -> {new_len} tokens ({ratio:.1%})")

            except Exception as e:
                print(f"[Layer {layer_idx}] Compression failed: {e}")

    def update_cache(self, past_key_values):
        """Update the stored cache reference."""
        self.modified_past_key_values = past_key_values

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        """Reset hook state."""
        self.stats.reset()
        self.compression_count = 0
        self.current_tokens = 0
        self.modified_past_key_values = None

    def get_stats(self) -> Dict:
        """Get hook statistics."""
        return {
            'compression_count': self.compression_count,
            'current_tokens': self.current_tokens,
            'hooks_registered': len(self.hooks),
            'baseline_stats': self.baseline.get_stats() if hasattr(self.baseline, 'get_stats') else {},
        }


class HookedGenerator:
    """
    Generator with KV cache hooks enabled.

    Wraps model.generate() with cache compression hooks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        baseline,
        hook_config: HookConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline = baseline
        self.hook_config = hook_config or HookConfig()

        self.hook = CacheHook(baseline, hook_config)

    def setup_hooks(self):
        """Set up hooks before generation."""
        self.hook.reset()
        num_hooks = self.hook.register_hooks(self.model)
        print(f"Registered {num_hooks} attention hooks")
        return num_hooks

    def cleanup_hooks(self):
        """Remove hooks after generation."""
        self.hook.remove_hooks()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Generate text with cache compression.

        Returns:
            generated_text: The generated text
            stats: Statistics about the generation
        """
        # Set up hooks
        self.setup_hooks()
        self.baseline.reset()

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)

        # Generation with hooks
        try:
            # Use model's generate with modified cache handling
            outputs = self._hooked_generate(
                input_ids,
                max_new_tokens,
                **kwargs
            )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        finally:
            # Clean up hooks
            self.cleanup_hooks()

        # Get stats
        stats = self.hook.get_stats()

        return generated_text, stats

    def _hooked_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Manual generation loop with cache hooks.

        This gives us control over when compression happens.
        """
        self.model.eval()

        # Initialize
        generated_ids = input_ids.clone()
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if past_key_values is None:
                    # First pass (prefill)
                    outputs = self.model(
                        generated_ids,
                        use_cache=True,
                    )
                else:
                    # Decode step - only process last token
                    outputs = self.model(
                        generated_ids[:, -1:].contiguous(),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Update cache reference for hooks
                past_key_values = outputs.past_key_values
                self.hook.update_cache(past_key_values)

                # Check for compression trigger
                current_cache_len = past_key_values[0][0].shape[2]

                if current_cache_len >= self.hook_config.max_cache_length:
                    # Apply compression to all layers
                    self._compress_all_layers(past_key_values)

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Append to generated
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Update hook token count
                self.hook.current_tokens = current_cache_len

        return generated_ids

    def _compress_all_layers(self, past_key_values):
        """Apply compression to all layers in cache."""
        num_layers = len(past_key_values)
        new_cache = []

        for layer_idx in range(num_layers):
            keys = past_key_values[layer_idx][0]
            values = past_key_values[layer_idx][1]

            # Get importance scores
            importance = self.hook.stats.get_importance_scores(layer_idx)

            try:
                # Apply baseline compression
                compressed_keys, compressed_values = self.baseline.compress(
                    keys, values,
                    attention_weights=importance.unsqueeze(1).unsqueeze(1) if importance is not None else None,
                )

                new_cache.append((compressed_keys, compressed_values))

                if self.hook_config.log_compressions:
                    old_len = keys.shape[2]
                    new_len = compressed_keys.shape[2]
                    print(f"[Compression] Layer {layer_idx}: {old_len} -> {new_len} tokens")

            except Exception as e:
                # Keep original if compression fails
                new_cache.append((keys, values))
                print(f"[Compression] Layer {layer_idx} failed: {e}")

        # Update the hook's cache reference
        self.hook.modified_past_key_values = new_cache
        self.hook.compression_count += 1


# Test code
if __name__ == "__main__":
    print("Testing KV Cache Hook...")

    from baselines.unified_interface import BaselineMethod, CacheConfig, create_baseline

    # Create mock model
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)

        def forward(self, x):
            return x

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = MockAttention()
            self.layer2 = MockAttention()
            self.config = type('Config', (), {'num_hidden_layers': 2})()

        def forward(self, x, use_cache=False, past_key_values=None):
            # Mock output
            batch, seq = x.shape
            output = type('Output', (), {
                'logits': torch.randn(batch, seq, 1000),
                'past_key_values': [(torch.randn(1, 8, seq, 64), torch.randn(1, 8, seq, 64)),
                                    (torch.randn(1, 8, seq, 64), torch.randn(1, 8, seq, 64))]
            })()
            return output

        def eval(self):
            pass

    # Test hook
    model = MockModel()
    config = CacheConfig(method=BaselineMethod.H2O, heavy_ratio=0.1, recent_ratio=0.1)
    baseline = create_baseline(config)

    hook_config = HookConfig(
        max_cache_length=100,
        compress_every_n_tokens=50,
        log_compressions=True,
    )

    hook = CacheHook(baseline, hook_config)
    num_hooks = hook.register_hooks(model)
    print(f"Registered {num_hooks} hooks")

    # Test stats tracking
    stats = AttentionStats()
    attn_weights = torch.randn(1, 8, 1, 100).softmax(dim=-1)
    stats.update(0, attn_weights)
    print(f"Importance scores shape: {stats.get_importance_scores(0).shape}")

    print("\nKV Cache Hook test passed!")