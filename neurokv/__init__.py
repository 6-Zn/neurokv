"""
NeuroKV - Adaptive Hierarchical Key-Value Cache Management

A learning-based KV cache management system for long-context LLM inference.
"""

from .policy import PolicyNetwork, PolicyNetworkSmall, PolicyFeatures
from .cache import ThreeTierCacheManager, TwoTierCacheManager, Quantizer
from .oracle import OracleAttribution, OracleTraceGenerator, OracleConfig, AttributionMethod
from .trainer import ImitationTrainer, TrainerConfig

__version__ = "0.1.0"

__all__ = [
    # Policy
    "PolicyNetwork",
    "PolicyNetworkSmall",
    "PolicyFeatures",
    # Cache
    "ThreeTierCacheManager",
    "TwoTierCacheManager",
    "Quantizer",
    # Oracle
    "OracleAttribution",
    "OracleTraceGenerator",
    "OracleConfig",
    "AttributionMethod",
    # Trainer
    "ImitationTrainer",
    "TrainerConfig",
]