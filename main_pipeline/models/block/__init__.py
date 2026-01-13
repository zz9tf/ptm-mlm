"""
Block module: Process embeddings/features.

Each block is in a separate file for clarity.
"""

from .lora import LoRABlock
from typing import Dict, Any
import torch.nn as nn


# Registry for blocks
BLOCK_REGISTRY = {
    "lora": LoRABlock,
}


def build_block(block_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Build block from type and configuration.
    
    @param block_type: Type of block ("linear", "lora", "mamba", etc.)
    @param config: Configuration dictionary
    @returns: Block instance
    """
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}. Available: {list(BLOCK_REGISTRY.keys())}")
    
    block_class = BLOCK_REGISTRY[block_type]
    # Remove 'type' from config as it's not a parameter for block classes
    config_clean = {k: v for k, v in config.items() if k != 'type'}
    cls = block_class(**config_clean)
    cls.type = block_type
    return cls


__all__ = [
    "LoRABlock",
    "build_block",
    "BLOCK_REGISTRY",
]

