"""
NeuroKV Policy Network

A small Transformer-based policy network that decides KV cache actions.
"""

import torch
import torch.nn as nn
import math


class PolicyNetwork(nn.Module):
    """
    Policy network for KV cache management.

    Input: per-token features (attention stats, position, layer-id, staleness)
    Output: action distribution over {KEEP_FP16, COMPRESS_INT4, OFFLOAD_DRAM, EVICT}

    Architecture:
    - 4 layers, hidden size 256, 4 heads (~8M parameters)
    - Shared across layers with layer-id embedding
    """

    # Action constants
    KEEP_FP16 = 0
    COMPRESS_INT4 = 1
    OFFLOAD_DRAM = 2
    EVICT = 3

    NUM_ACTIONS = 4

    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_positions: int = 131072,  # 128K context
        num_llm_layers: int = 32,  # Number of layers in target LLM
        feature_dim: int = 16,  # Input feature dimension
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_llm_layers = num_llm_layers
        self.feature_dim = feature_dim

        # Layer ID embedding (for shared policy across LLM layers)
        self.layer_embedding = nn.Embedding(num_llm_layers, hidden_size)

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_size)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_positions, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: action distribution
        self.output_head = nn.Linear(hidden_size, self.NUM_ACTIONS)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (batch, seq_len, feature_dim) - per-token features
            positions: (batch, seq_len) - absolute positions
            layer_ids: (batch,) - layer ID for each batch element

        Returns:
            action_logits: (batch, seq_len, NUM_ACTIONS) - action distribution logits
        """
        batch_size, seq_len, _ = features.shape

        # Project features
        x = self.input_proj(features)  # (batch, seq_len, hidden_size)

        # Add positional embedding
        pos_emb = self.pos_embedding(positions)  # (batch, seq_len, hidden_size)
        x = x + pos_emb

        # Add layer embedding (broadcast across sequence)
        layer_emb = self.layer_embedding(layer_ids)  # (batch, hidden_size)
        x = x + layer_emb.unsqueeze(1)  # (batch, seq_len, hidden_size)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden_size)

        # Output head
        action_logits = self.output_head(x)  # (batch, seq_len, NUM_ACTIONS)

        return action_logits

    def get_action_probs(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        return torch.softmax(action_logits, dim=-1)

    def sample_actions(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Sample actions from the policy."""
        probs = self.get_action_probs(action_logits)
        actions = torch.multinomial(probs.view(-1, self.NUM_ACTIONS), num_samples=1)
        return actions.view(action_logits.shape[0], action_logits.shape[1])

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class PolicyFeatures:
    """
    Feature extraction for policy network input.

    Features per token:
    1. Recent attention statistics (mean, max, std over last W steps)
    2. Position features (absolute, relative to current query)
    3. Layer features (layer ID)
    4. Staleness (time since last attended)
    5. Token type embedding
    """

    FEATURE_DIM = 16

    @staticmethod
    def extract_features(
        attention_weights: torch.Tensor,
        current_position: int,
        layer_id: int,
        history_window: int = 32,
        attention_history: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract features for each cached token.

        Args:
            attention_weights: (batch, heads, 1, seq_len) - current attention weights
            current_position: int - current generation position
            layer_id: int - layer index
            history_window: int - window for attention statistics
            attention_history: (batch, seq_len) - accumulated attention scores

        Returns:
            features: (batch, seq_len, FEATURE_DIM)
        """
        batch_size, num_heads, _, seq_len = attention_weights.shape

        # Attention statistics
        attn_mean = attention_weights.mean(dim=(1, 2)).squeeze(1)  # (batch, seq_len)
        attn_max = attention_weights.max(dim=1)[0].max(dim=1)[0].squeeze(1)  # (batch, seq_len)

        # Position features
        positions = torch.arange(seq_len, device=attention_weights.device)
        relative_pos = current_position - positions  # distance to current query

        # Staleness (use attention history if available)
        if attention_history is not None:
            staleness = current_position - attention_history.argmax(dim=-1)
        else:
            staleness = torch.zeros(seq_len, device=attention_weights.device)

        # Normalize features
        features = torch.zeros(batch_size, seq_len, PolicyFeatures.FEATURE_DIM,
                               device=attention_weights.device)

        # Fill feature slots
        features[:, :, 0] = attn_mean  # Mean attention
        features[:, :, 1] = attn_max   # Max attention
        features[:, :, 2] = positions.float() / 10000  # Normalized position
        features[:, :, 3] = relative_pos.float() / 100  # Relative position
        features[:, :, 4] = layer_id / 32  # Normalized layer ID
        features[:, :, 5] = staleness.float() / 100  # Staleness

        # Additional features from history if available
        if attention_history is not None:
            features[:, :, 6] = attention_history  # Accumulated attention
            features[:, :, 7] = attention_history / (current_position + 1)  # Normalized accumulated

        return features

    @staticmethod
    def extract_from_attention(
        attention_weights: torch.Tensor,
        current_position: int,
        layer_id: int,
        importance_scores: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Wrapper for extract_features that uses importance scores as history.

        Args:
            attention_weights: (batch, heads, 1, seq_len)
            current_position: int
            layer_id: int
            importance_scores: (batch, seq_len) - accumulated importance

        Returns:
            features: (batch, seq_len, FEATURE_DIM)
        """
        return PolicyFeatures.extract_features(
            attention_weights,
            current_position,
            layer_id,
            history_window=32,
            attention_history=importance_scores,
        )


class PolicyNetworkSmall(PolicyNetwork):
    """
    Smaller version of policy network for initial experiments.

    Architecture: 2 layers, 128 hidden, 2 heads (~1M parameters)
    """

    def __init__(self, num_llm_layers: int = 32, feature_dim: int = 16):
        super().__init__(
            num_layers=2,
            hidden_size=128,
            num_heads=2,
            dropout=0.1,
            num_llm_layers=num_llm_layers,
            feature_dim=feature_dim,
        )


# Test code
if __name__ == "__main__":
    # Create policy network
    policy = PolicyNetwork()
    print(f"Full policy network: {policy.get_num_parameters()} parameters")

    # Create small policy
    policy_small = PolicyNetworkSmall()
    print(f"Small policy network: {policy_small.get_num_parameters()} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 100
    features = torch.randn(batch_size, seq_len, PolicyFeatures.FEATURE_DIM)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    layer_ids = torch.randint(0, 32, (batch_size,))

    action_logits = policy(features, positions, layer_ids)
    print(f"Action logits shape: {action_logits.shape}")

    # Test action sampling
    actions = policy.sample_actions(action_logits)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Action distribution: {torch.bincount(actions.flatten(), minlength=4)}")