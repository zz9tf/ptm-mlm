"""
Models package for PTM-MLM.

Simple and direct structure:
- Blocks: Process embeddings (LinearBlock, LoRABlock, MambaBlock, etc.)
- Heads: Generate sequences (OriginalHead, PTMHead)
- Model: Main model (PTMModel)
"""

from .block import (
    LoRABlock,
    build_block,
    BLOCK_REGISTRY,
)
from .head import (
    OriginalHead,
    PTMHead,
    FunctionalRoleHead,
    build_head,
    HEAD_REGISTRY,
)
from .model import PTMModel

__all__ = [
    # Blocks
    "LoRABlock",
    "build_block",
    "BLOCK_REGISTRY",
    # Heads
    "OriginalHead",
    "PTMHead",
    "FunctionalRoleHead",
    "build_head",
    "HEAD_REGISTRY",
    # Main model
    "PTMModel",
]
