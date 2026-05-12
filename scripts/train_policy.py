#!/usr/bin/env python3
"""
Train Policy Network on Real Oracle Traces

Day 6: Train the NeuroKV policy network using real attention-based oracle traces.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import json
import argparse
from datetime import datetime

from neurokv.trainer.imitation import ImitationTrainer, TrainerConfig
from neurokv.policy import PolicyNetworkSmall


def load_json_traces(path: str) -> list:
    """Load oracle traces from JSON file."""
    with open(path, 'r') as f:
        traces = json.load(f)
    print(f"Loaded {len(traces)} traces from {path}")
    return traces


def extract_dataset_from_json_traces(traces: list, max_positions: int = 131072) -> tuple:
    """
    Convert JSON traces to training tensors.

    The JSON traces have:
    - steps: list of {phase, importances, labels, features} dicts
    - importances/labels/features are lists of lists (from tensors)

    Returns:
        features: (total_samples, feature_dim) tensor
        labels: (total_samples,) tensor
    """
    all_features = []
    all_labels = []
    all_positions = []
    all_layer_ids = []

    for trace_idx, trace in enumerate(traces):
        num_layers = trace.get('num_layers', 22)  # Qwen2.5-0.5B has 22 layers

        for step_idx, step in enumerate(trace['steps']):
            phase = step['phase']

            # Get importances, labels, features for each layer in this step
            importances = step.get('importances', [])
            labels = step.get('labels', [])
            features = step.get('features', [])

            # Each is a list of tensors (one per layer)
            for layer_idx, (feat, lab) in enumerate(zip(features, labels)):
                # Skip if empty
                if not feat or not lab:
                    continue

                # Convert to tensors - feat is nested list (batch, seq_len, feature_dim)
                feat_tensor = torch.tensor(feat) if not isinstance(feat, torch.Tensor) else feat
                lab_tensor = torch.tensor(lab) if not isinstance(lab, torch.Tensor) else lab

                # Handle different shapes
                # feat_tensor expected shape: (batch, seq_len, feature_dim) or (seq_len, feature_dim)
                if feat_tensor.dim() == 3:
                    feat_flat = feat_tensor.reshape(-1, feat_tensor.shape[-1])
                    lab_flat = lab_tensor.reshape(-1)
                elif feat_tensor.dim() == 2:
                    feat_flat = feat_tensor
                    lab_flat = lab_tensor if lab_tensor.dim() == 1 else lab_tensor.flatten()
                elif feat_tensor.dim() == 1:
                    # Skip invalid shapes
                    continue
                else:
                    continue

                # Generate positions - clamp to max_positions
                seq_len = feat_flat.shape[0]
                positions = torch.arange(seq_len).clamp_max(max_positions - 1)

                # Generate layer ids - clamp to num_layers-1
                layer_ids = torch.full((seq_len,), min(layer_idx, num_layers - 1), dtype=torch.long)

                all_features.append(feat_flat)
                all_labels.append(lab_flat.long())
                all_positions.append(positions)
                all_layer_ids.append(layer_ids)

    if not all_features:
        raise ValueError("No valid training data found in traces")

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    positions = torch.cat(all_positions, dim=0)
    layer_ids = torch.cat(all_layer_ids, dim=0)

    return features, labels, positions, layer_ids


def compute_label_distribution(labels: torch.Tensor) -> dict:
    """Compute distribution of action labels."""
    counts = torch.bincount(labels, minlength=4).tolist()
    total = sum(counts)
    return {
        'KEEP_FP16': counts[0],
        'COMPRESS_INT4': counts[1],
        'OFFLOAD_DRAM': counts[2],
        'EVICT': counts[3],
        'total': total,
        'percentages': {k: f"{v/total*100:.1f}%" for k, v in zip(['KEEP', 'COMPRESS', 'OFFLOAD', 'EVICT'], counts)},
    }


def main():
    parser = argparse.ArgumentParser(description="Train policy network on oracle traces")
    parser.add_argument('--traces', type=str, default='data/oracle_traces.json',
                        help='Path to oracle traces JSON')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='checkpoints/policy_trained.pt',
                        help='Output checkpoint path')
    parser.add_argument('--small', action='store_true',
                        help='Use small policy network (1M params)')
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroKV Policy Network Training - Day 6")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load traces
    print(f"\nLoading traces from {args.traces}...")
    traces = load_json_traces(args.traces)

    # Get num_layers from first trace
    num_llm_layers = traces[0].get('num_layers', 22) if traces else 22
    print(f"LLM layers from traces: {num_llm_layers}")

    # Extract dataset
    print("\nExtracting training dataset...")
    features, labels, positions, layer_ids = extract_dataset_from_json_traces(traces)
    print(f"Total samples: {features.shape[0]}")
    print(f"Feature dimension: {features.shape[1]}")

    # Show label distribution
    dist = compute_label_distribution(labels)
    print("\nLabel distribution:")
    print(f"  KEEP_FP16 (Tier-1): {dist['KEEP_FP16']} ({dist['percentages']['KEEP']})")
    print(f"  COMPRESS_INT4 (Tier-2): {dist['COMPRESS_INT4']} ({dist['percentages']['COMPRESS']})")
    print(f"  OFFLOAD_DRAM (Tier-3): {dist['OFFLOAD_DRAM']} ({dist['percentages']['OFFLOAD']})")
    print(f"  EVICT: {dist['EVICT']} ({dist['percentages']['EVICT']})")

    # Split into train/val (90/10)
    num_samples = features.shape[0]
    split_idx = int(num_samples * 0.9)

    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    train_positions = positions[:split_idx]
    train_layer_ids = layer_ids[:split_idx]

    val_features = features[split_idx:]
    val_labels = labels[split_idx:]
    val_positions = positions[split_idx:]
    val_layer_ids = layer_ids[split_idx:]

    print(f"\nTrain samples: {train_features.shape[0]}")
    print(f"Val samples: {val_features.shape[0]}")

    # Configure trainer
    if args.small:
        config = TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_layers=2,
            hidden_size=128,
            num_heads=2,
            num_llm_layers=num_llm_layers,
            feature_dim=features.shape[1],
            checkpoint_dir=os.path.dirname(args.output) or 'checkpoints',
        )
    else:
        config = TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_layers=4,
            hidden_size=256,
            num_heads=4,
            num_llm_layers=num_llm_layers,
            feature_dim=features.shape[1],
            checkpoint_dir=os.path.dirname(args.output) or 'checkpoints',
        )

    print(f"\nTrainer config:")
    print(f"  Network: {config.num_layers} layers, {config.hidden_size} hidden, {config.num_heads} heads")
    print(f"  Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.learning_rate}")

    # Create trainer
    trainer = ImitationTrainer(config)
    print(f"\nPolicy network parameters: {trainer.policy.get_num_parameters():,}")

    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    metrics = trainer.train(
        train_features,
        train_labels,
        train_positions,
        train_layer_ids,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Final accuracy: {metrics['final_accuracy']:.2%}")
    print(f"Best loss: {metrics['best_loss']:.4f}")
    print(f"Epochs trained: {metrics['epochs_trained']}")

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Evaluation")
    print("=" * 60)

    val_metrics = trainer.evaluate(val_features, val_labels)
    print(f"Validation loss: {val_metrics['loss']:.4f}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.2%}")
    print("Per-class accuracy:")
    action_names = ['KEEP_FP16', 'COMPRESS_INT4', 'OFFLOAD_DRAM', 'EVICT']
    for i, name in enumerate(action_names):
        key = f"acc_class_{i}"
        if key in val_metrics:
            print(f"  {name}: {val_metrics[key]:.2%}")

    # Save final checkpoint
    trainer.save_checkpoint("final")
    print(f"\nSaved checkpoint to {config.checkpoint_dir}/policy_final.pt")

    print("\n" + "=" * 60)
    print("Day 6 Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()