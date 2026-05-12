#!/usr/bin/env python3
"""
Binary Policy Training - Day 7

Simplify to binary classification: KEEP vs EVICT.
This addresses the COMPRESS_INT4 prediction difficulty from Day 6.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
from datetime import datetime


class BinaryPolicyNetwork(nn.Module):
    """
    Binary policy network: KEEP (0) vs EVICT (1).

    Simpler architecture for better learning on limited data.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_llm_layers: int = 24,
        feature_dim: int = 16,
        max_positions: int = 131072,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_llm_layers = num_llm_layers

        # Layer ID embedding
        self.layer_embedding = nn.Embedding(num_llm_layers, hidden_size)

        # Position embedding
        self.pos_embedding = nn.Embedding(max_positions, hidden_size)

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_size)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # Binary output
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features, positions, layer_ids):
        """
        Args:
            features: (batch, seq_len, feature_dim)
            positions: (batch, seq_len)
            layer_ids: (batch,)

        Returns:
            logits: (batch, seq_len, 2)
        """
        # Project features
        x = self.feature_proj(features)

        # Add position embedding
        x = x + self.pos_embedding(positions)

        # Add layer embedding
        x = x + self.layer_embedding(layer_ids).unsqueeze(1)

        # MLP
        logits = self.mlp(x)

        return logits

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def load_and_convert_to_binary(traces_path: str, keep_ratio: float = 0.15):
    """
    Load traces and convert to binary labels.

    KEEP (0): top keep_ratio tokens by importance
    EVICT (1): remaining tokens
    """
    with open(traces_path, 'r') as f:
        traces = json.load(f)

    print(f"Loaded {len(traces)} traces from {traces_path}")

    all_features = []
    all_labels = []
    all_positions = []
    all_layer_ids = []

    for trace in traces:
        num_layers = trace.get('num_layers', 24)

        for step in trace['steps']:
            features = step.get('features', [])
            labels = step.get('labels', [])
            importances = step.get('importances', [])

            for layer_idx, (feat, lab, imp) in enumerate(zip(features, labels, importances)):
                if not feat or not lab:
                    continue

                feat_tensor = torch.tensor(feat)
                imp_tensor = torch.tensor(imp)

                if feat_tensor.dim() == 3:
                    feat_flat = feat_tensor.reshape(-1, feat_tensor.shape[-1])
                    imp_flat = imp_tensor.reshape(-1)
                elif feat_tensor.dim() == 2:
                    feat_flat = feat_tensor
                    imp_flat = imp_tensor
                else:
                    continue

                seq_len = feat_flat.shape[0]

                # Convert to binary: top keep_ratio importance -> KEEP (0), rest -> EVICT (1)
                # Use importance scores to determine keep threshold
                imp_threshold = torch.quantile(imp_flat, 1 - keep_ratio)
                binary_labels = (imp_flat <= imp_threshold).long()  # 0=KEEP (high importance), 1=EVICT

                positions = torch.arange(seq_len).clamp_max(131071)
                layer_ids = torch.full((seq_len,), min(layer_idx, num_layers - 1), dtype=torch.long)

                all_features.append(feat_flat)
                all_labels.append(binary_labels)
                all_positions.append(positions)
                all_layer_ids.append(layer_ids)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    positions = torch.cat(all_positions, dim=0)
    layer_ids = torch.cat(all_layer_ids, dim=0)

    return features, labels, positions, layer_ids, traces[0].get('num_layers', 24)


def main():
    parser = argparse.ArgumentParser(description="Binary policy training")
    parser.add_argument('--traces', type=str, default='data/oracle_traces_extended.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--keep-ratio', type=float, default=0.15,
                        help='Ratio of tokens to KEEP (top importance)')
    parser.add_argument('--output', type=str, default='checkpoints/binary_policy.pt')
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroKV Binary Policy Training - Day 7")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and convert
    features, labels, positions, layer_ids, num_layers = load_and_convert_to_binary(
        args.traces, args.keep_ratio
    )

    print(f"\nTotal samples: {features.shape[0]}")
    print(f"Feature dimension: {features.shape[1]}")

    # Label distribution
    keep_count = (labels == 0).sum().item()
    evict_count = (labels == 1).sum().item()
    total = labels.shape[0]
    print(f"\nBinary label distribution:")
    print(f"  KEEP (high importance): {keep_count} ({keep_count/total*100:.1f}%)")
    print(f"  EVICT: {evict_count} ({evict_count/total*100:.1f}%)")

    # Split train/val
    split_idx = int(features.shape[0] * 0.9)
    train_features, train_labels, train_positions, train_layer_ids = \
        features[:split_idx], labels[:split_idx], positions[:split_idx], layer_ids[:split_idx]
    val_features, val_labels, val_positions, val_layer_ids = \
        features[split_idx:], labels[split_idx:], positions[split_idx:], layer_ids[split_idx:]

    print(f"Train: {train_features.shape[0]}, Val: {val_features.shape[0]}")

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BinaryPolicyNetwork(
        hidden_size=128,
        num_llm_layers=num_layers,
        feature_dim=features.shape[1],
    ).to(device)

    print(f"\nModel parameters: {model.get_num_parameters():,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_features, train_labels, train_positions, train_layer_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (bf, bl, bp, bli) in enumerate(train_loader):
            bf = bf.to(device).unsqueeze(1)  # (batch, 1, feat_dim)
            bl = bl.to(device)
            bp = bp.to(device).unsqueeze(1)
            bli = bli.to(device)

            logits = model(bf, bp, bli).squeeze(1)  # (batch, 2)
            loss = criterion(logits, bl)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * bf.shape[0]
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == bl).sum().item()
            epoch_total += bl.shape[0]

        avg_loss = epoch_loss / epoch_total
        accuracy = epoch_correct / epoch_total

        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {accuracy:.2%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'config': {
                    'hidden_size': 128,
                    'num_llm_layers': num_layers,
                    'feature_dim': features.shape[1],
                }
            }, args.output)
            print(f"  Saved checkpoint to {args.output}")

    # Validation
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        keep_correct = 0
        keep_total = 0
        evict_correct = 0
        evict_total = 0

        for start in range(0, val_features.shape[0], args.batch_size):
            end = min(start + args.batch_size, val_features.shape[0])
            bf = val_features[start:end].to(device).unsqueeze(1)
            bl = val_labels[start:end].to(device)
            bp = val_positions[start:end].to(device).unsqueeze(1)
            bli = val_layer_ids[start:end].to(device)

            logits = model(bf, bp, bli).squeeze(1)
            preds = logits.argmax(dim=-1)

            val_correct += (preds == bl).sum().item()
            val_total += bl.shape[0]

            # Per-class
            keep_mask = bl == 0
            keep_correct += (preds[keep_mask] == 0).sum().item()
            keep_total += keep_mask.sum().item()

            evict_mask = bl == 1
            evict_correct += (preds[evict_mask] == 1).sum().item()
            evict_total += evict_mask.sum().item()

    print(f"Overall accuracy: {val_correct/val_total:.2%}")
    print(f"KEEP accuracy: {keep_correct/keep_total:.2%}")
    print(f"EVICT accuracy: {evict_correct/evict_total:.2%}")

    print("\n" + "=" * 60)
    print("Day 7 Binary Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()