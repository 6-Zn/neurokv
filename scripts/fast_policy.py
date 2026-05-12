#!/usr/bin/env python3
"""
Fast Policy Inference - Day 9

Use trained policy weights with optimized batched inference.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import time
from datetime import datetime

from baselines.unified_interface import (
    get_cache_layers,
    get_layer_kv,
    set_layer_kv,
    get_cache_length,
)


class FastPolicyHandler:
    """
    Fast policy handler using trained weights with batched inference.
    """

    def __init__(self, policy_path='checkpoints/binary_policy.pt', device='cuda'):
        self.device = device

        # Load trained policy
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

        self.keep_ratio = 0.15
        self.stats = {'method': 'NeuroKV-Fast', 'compressions': 0, 'inference_time_ms': 0}

    def reset(self):
        self.stats['compressions'] = 0
        self.stats['inference_time_ms'] = 0

    def compress_batched(self, cache, current_position):
        """
        Batched compression - process all layers together.
        """
        layers = get_cache_layers(cache)
        if layers is None:
            return 0

        start_time = time.perf_counter()

        # Collect all data
        all_features = []
        all_positions = []
        all_layer_ids = []
        layer_seq_lens = []

        for layer_idx, layer in enumerate(layers):
            keys, values = get_layer_kv(layer)
            if keys is None:
                layer_seq_lens.append(0)
                continue

            seq_len = keys.shape[2]
            layer_seq_lens.append(seq_len)

            features = self._extract_features(keys, values, seq_len, layer_idx, current_position)
            positions = torch.arange(seq_len, device=self.device).clamp_max(131071)
            layer_ids = torch.full((seq_len,), layer_idx, device=self.device, dtype=torch.long)

            all_features.append(features)
            all_positions.append(positions)
            all_layer_ids.append(layer_ids)

        if not all_features:
            return 0

        # Concatenate all
        all_features = torch.cat(all_features, dim=0)  # (total_tokens, feat_dim)
        all_positions = torch.cat(all_positions, dim=0)
        all_layer_ids = torch.cat(all_layer_ids, dim=0)

        # Single batched forward pass
        with torch.no_grad():
            # Reshape for policy (batch=1, seq_len, feat_dim)
            total_tokens = all_features.shape[0]
            features_batch = all_features.unsqueeze(0)
            positions_batch = all_positions.unsqueeze(0)
            layer_ids_batch = all_layer_ids[:1]  # Use first layer_id for batch

            logits = self.policy(features_batch, positions_batch, layer_ids_batch)
            predictions = logits.squeeze(0).argmax(dim=-1)  # (total_tokens,)

        inference_time = (time.perf_counter() - start_time) * 1000
        self.stats['inference_time_ms'] += inference_time

        # Apply predictions to each layer
        total_evicted = 0
        pred_offset = 0

        for layer_idx, seq_len in enumerate(layer_seq_lens):
            if seq_len == 0:
                continue

            layer_preds = predictions[pred_offset:pred_offset + seq_len]
            pred_offset += seq_len

            # Always keep attention sinks and recent
            sink_size = 4
            recent_size = max(32, int(seq_len * 0.05))

            sink_indices = torch.arange(0, sink_size, device=self.device)
            recent_indices = torch.arange(seq_len - recent_size, seq_len, device=self.device)

            # From middle, keep those predicted as KEEP
            eval_start = sink_size
            eval_end = seq_len - recent_size

            if eval_end > eval_start:
                middle_preds = layer_preds[eval_start:eval_end]
                middle_keep = (middle_preds == 0).nonzero(as_tuple=True)[0] + eval_start
                keep_indices = torch.unique(torch.cat([sink_indices, middle_keep, recent_indices]))
            else:
                keep_indices = torch.unique(torch.cat([sink_indices, recent_indices]))

            # Minimum keep ratio
            min_keep = max(sink_size + recent_size, int(seq_len * self.keep_ratio))
            if len(keep_indices) < min_keep:
                needed = min_keep - len(keep_indices)
                extra = torch.arange(max(0, seq_len - recent_size - needed), seq_len - recent_size, device=self.device)
                keep_indices = torch.unique(torch.cat([keep_indices, extra]))

            # Apply compression
            if len(keep_indices) < seq_len:
                keys, values = get_layer_kv(layers[layer_idx])
                new_keys = keys[:, :, keep_indices, :]
                new_values = values[:, :, keep_indices, :]
                set_layer_kv(layers[layer_idx], new_keys, new_values)
                total_evicted += seq_len - len(keep_indices)

        self.stats['compressions'] += 1
        return total_evicted

    def _extract_features(self, keys, values, seq_len, layer_idx, current_position):
        features = torch.zeros(seq_len, self.config['feature_dim'], device=self.device)

        # Position
        features[:, 0] = torch.arange(seq_len, device=self.device).float() / 10000
        features[:, 1] = (current_position - torch.arange(seq_len, device=self.device)).float() / 100

        # Layer
        features[:, 2] = layer_idx / self.config['num_llm_layers']

        # KV norms
        if keys is not None and keys.shape[0] > 0:
            key_norm = keys[0].norm(dim=-1).mean(dim=0)
            value_norm = values[0].norm(dim=-1).mean(dim=0)
            key_max = key_norm.max() + 1e-8
            value_max = value_norm.max() + 1e-8
            features[:, 3] = key_norm / key_max
            features[:, 4] = value_norm / value_max

        # Log distance
        features[:, 5] = torch.log1p(torch.clamp(current_position - torch.arange(seq_len, device=self.device), min=0).float())

        return features

    def get_stats(self):
        return self.stats


def test_inference_speed():
    """Test batched inference speed."""
    print("=" * 60)
    print("Fast Policy Inference Speed Test")
    print("=" * 60)

    handler = FastPolicyHandler()
    print(f"Policy parameters: {handler.policy.get_num_parameters():,}")

    # Create dummy cache
    batch, heads, head_dim = 1, 14, 64
    seq_lengths = [100, 500, 1000, 2000, 4000]

    print("\nInference time by sequence length:")
    print("-" * 50)

    for seq_len in seq_lengths:
        keys = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        values = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

        class DummyCache:
            def __init__(self, layers):
                self.layers = layers

        cache_layers = []
        for _ in range(24):
            cache_layers.append([keys.clone(), values.clone()])

        cache = DummyCache(cache_layers)

        # Warmup
        handler.reset()
        handler.compress_batched(cache, seq_len - 1)

        # Measure
        handler.reset()
        torch.cuda.synchronize()

        times = []
        for _ in range(3):
            start = time.perf_counter()
            handler.compress_batched(cache, seq_len - 1)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        print(f"  Seq {seq_len}: {avg_time:.2f}ms total, {handler.stats['inference_time_ms']:.2f}ms policy")

    print("\n" + "=" * 60)
    print("Speed Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_inference_speed()