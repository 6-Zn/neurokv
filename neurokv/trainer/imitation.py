"""
Imitation Learning Trainer for NeuroKV

Stage 1: Train policy network to match oracle actions via behavior cloning.

Trains the Transformer policy to predict oracle decisions:
- Input: per-token features (attention stats, position, layer, staleness)
- Output: action distribution (KEEP_FP16, COMPRESS_INT4, OFFLOAD_DRAM, EVICT)

Uses standard cross-entropy loss on oracle action labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import time


@dataclass
class TrainerConfig:
    """Configuration for imitation learning training."""
    # Training parameters
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Policy network parameters
    num_layers: int = 2        # 2 for small, 4 for full
    hidden_size: int = 128     # 128 for small, 256 for full
    num_heads: int = 2         # 2 for small, 4 for full
    num_llm_layers: int = 32   # Number of layers in target LLM
    feature_dim: int = 16

    # Training settings
    gradient_clip: float = 1.0
    early_stopping_patience: int = 3
    checkpoint_dir: str = "checkpoints/"
    log_interval: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ImitationTrainer:
    """
    Behavior Cloning trainer for NeuroKV policy network.

    Trains policy to match oracle decisions via cross-entropy loss.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config

        # Initialize policy network
        from neurokv.policy import PolicyNetwork, PolicyNetworkSmall

        if config.num_layers == 2 and config.hidden_size == 128:
            self.policy = PolicyNetworkSmall(
                num_llm_layers=config.num_llm_layers,
                feature_dim=config.feature_dim,
            )
        else:
            self.policy = PolicyNetwork(
                num_layers=config.num_layers,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_llm_layers=config.num_llm_layers,
                feature_dim=config.feature_dim,
            )

        self.policy = self.policy.to(config.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

        # Metrics
        self.train_losses = []
        self.train_accuracies = []

    def train(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Train policy network on oracle traces.

        Args:
            features: (num_samples, feature_dim) - input features
            labels: (num_samples,) - oracle action labels
            positions: (num_samples,) - token positions (optional)
            layer_ids: (num_samples,) - layer IDs (optional)

        Returns:
            metrics: Training metrics dictionary
        """
        # Move to device
        features = features.to(self.config.device)
        labels = labels.to(self.config.device)

        # Generate positions and layer_ids if not provided
        num_samples = features.shape[0]
        if positions is None:
            positions = torch.zeros(num_samples, dtype=torch.long, device=self.config.device)
        else:
            positions = positions.to(self.config.device)

        if layer_ids is None:
            layer_ids = torch.zeros(num_samples, dtype=torch.long, device=self.config.device)
        else:
            layer_ids = layer_ids.to(self.config.device)

        # Create dataset
        dataset = TensorDataset(features, labels, positions, layer_ids)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Training loop
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Dataset size: {num_samples} samples")
        print(f"Policy network: {self.policy.get_num_parameters():,} parameters")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (batch_features, batch_labels, batch_positions, batch_layer_ids) in enumerate(dataloader):
                # Forward pass
                # Note: policy expects (batch, seq_len, feature_dim)
                # We're treating each token as a single sequence of length 1
                batch_features = batch_features.unsqueeze(1)  # (batch, 1, feature_dim)
                batch_positions = batch_positions.unsqueeze(1)  # (batch, 1)

                action_logits = self.policy(batch_features, batch_positions, batch_layer_ids)
                action_logits = action_logits.squeeze(1)  # (batch, num_actions)

                # Compute loss - CrossEntropy expects (batch, num_classes) logits and (batch,) targets
                loss = self.criterion(action_logits, batch_labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item() * batch_features.shape[0]
                predictions = action_logits.argmax(dim=-1)
                epoch_correct += (predictions == batch_labels).sum().item()
                epoch_total += batch_labels.shape[0]

                self.global_step += 1

                # Log
                if batch_idx % self.config.log_interval == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {(predictions == batch_labels).float().mean().item():.2%}")

            # Epoch metrics
            avg_loss = epoch_loss / epoch_total
            accuracy = epoch_correct / epoch_total

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2%}")

            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
                self.save_checkpoint("best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "final_loss": avg_loss,
            "final_accuracy": accuracy,
            "best_loss": self.best_loss,
            "epochs_trained": self.current_epoch + 1,
        }

    def train_on_traces(
        self,
        traces: List[Dict],
    ) -> Dict[str, float]:
        """
        Train on oracle trace generator output.

        Args:
            traces: List of trace dictionaries from OracleTraceGenerator

        Returns:
            metrics: Training metrics
        """
        from neurokv.oracle import OracleTraceGenerator

        # Create temporary generator to extract dataset
        generator = OracleTraceGenerator.__new__(OracleTraceGenerator)
        generator.traces = traces

        features, labels = generator.get_dataset()

        return self.train(features, labels)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        path = os.path.join(self.config.checkpoint_dir, f"policy_{name}.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }, path)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best loss: {self.best_loss:.4f}")

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Evaluate policy on test data.

        Args:
            features: (num_samples, feature_dim)
            labels: (num_samples,)
            batch_size: Batch size for evaluation (to avoid memory issues)

        Returns:
            metrics: Evaluation metrics
        """
        features = features.to(self.config.device)
        labels = labels.to(self.config.device)

        self.policy.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_correct = [0, 0, 0, 0]
        class_total = [0, 0, 0, 0]

        with torch.no_grad():
            num_samples = features.shape[0]
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)

                batch_features = features[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                # Reshape for policy
                batch_features = batch_features.unsqueeze(1)
                positions = torch.zeros(batch_features.shape[0], dtype=torch.long, device=self.config.device).unsqueeze(1)
                layer_ids = torch.zeros(batch_features.shape[0], dtype=torch.long, device=self.config.device)

                action_logits = self.policy(batch_features, positions, layer_ids)
                action_logits = action_logits.squeeze(1)

                loss = self.criterion(action_logits, batch_labels)
                predictions = action_logits.argmax(dim=-1)

                total_loss += loss.item() * batch_features.shape[0]
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.shape[0]

                # Per-class stats
                for class_id in range(4):
                    mask = batch_labels == class_id
                    class_total[class_id] += mask.sum().item()
                    class_correct[class_id] += (predictions[mask] == class_id).sum().item()

        self.policy.train()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        class_accuracies = {}
        for class_id in range(4):
            if class_total[class_id] > 0:
                class_accuracies[f"acc_class_{class_id}"] = class_correct[class_id] / class_total[class_id]

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            **class_accuracies,
        }

    def get_policy(self) -> nn.Module:
        """Return trained policy network."""
        return self.policy


# Test code
if __name__ == "__main__":
    print("Testing Imitation Learning Trainer...")

    # Create trainer config
    config = TrainerConfig(
        epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        num_layers=2,
        hidden_size=128,
        num_heads=2,
    )

    trainer = ImitationTrainer(config)

    # Generate synthetic training data
    from neurokv.oracle import OracleTraceGenerator, OracleConfig, AttributionMethod

    oracle_config = OracleConfig(
        method=AttributionMethod.ATTENTION_ACCUMULATED,
        tier1_ratio=0.15,
        tier2_ratio=0.35,
    )

    generator = OracleTraceGenerator(oracle_config)

    # Generate multiple traces for training
    print("Generating training traces...")
    for _ in range(5):
        generator.generate_trace(
            prompt_length=100,
            generation_length=50,
            num_layers=4,
        )

    features, labels = generator.get_dataset()
    print(f"Training data: {features.shape[0]} samples")

    # Train
    metrics = trainer.train(features, labels)

    print(f"\nTraining completed!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Final accuracy: {metrics['final_accuracy']:.2%}")
    print(f"  Best loss: {metrics['best_loss']:.4f}")

    # Evaluate
    print("\nEvaluating on new trace...")
    generator.traces = []
    generator.generate_trace(prompt_length=100, generation_length=50, num_layers=4)
    test_features, test_labels = generator.get_dataset()

    eval_metrics = trainer.evaluate(test_features, test_labels)
    print(f"Test accuracy: {eval_metrics['accuracy']:.2%}")
    print(f"Per-class accuracy:")
    for k, v in eval_metrics.items():
        if k.startswith("acc_class"):
            print(f"  {k}: {v:.2%}")

    print("\nAll trainer tests passed!")