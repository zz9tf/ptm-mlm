"""
Head module: Generate sequences from processed features.

Each head is in a separate file for clarity.
"""

from models.head.original import OriginalHead
from models.head.ptm import PTMHead
from models.head.functional_role import FunctionalRoleHead
from typing import Dict, Any


# Registry for heads
HEAD_REGISTRY = {
    "original": OriginalHead,
    "ptm": PTMHead,
    "functional_role": FunctionalRoleHead,
}


def build_head(head_cfg: Dict[str, Any], d_model: int, vocab_size: int):
    """
    Build head from type and configuration.
    
    @param head_cfg: Configuration for head, should contain:
        - 'type': Type used to lookup in HEAD_REGISTRY and as key in model.heads (e.g., "original", "ptm")
        - 'weight': Optional weight for loss computation (default: 1.0)
        - Other parameters passed to head class
    @param d_model: Model dimension
    @param vocab_size: Vocabulary size
    @returns: Head instance
    """
    # Get type from config (required)
    head_type = head_cfg.get('type', None)
    if head_type is None:
        raise ValueError(
            f"'type' is required in head_cfg. Available types: {list(HEAD_REGISTRY.keys())}"
        )
    
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}. Available: {list(HEAD_REGISTRY.keys())}")
    
    # Get head class from registry
    head_class = HEAD_REGISTRY[head_type]
    
    # Remove 'type' and 'weight' from config as they're not parameters for head classes
    # 'type' is used for registry lookup, 'weight' is set as attribute after creation
    head_cfg_clean = {k: v for k, v in head_cfg.items() if k not in ('type', 'weight')}
    
    # Create head instance
    cls = head_class(d_model=d_model, vocab_size=vocab_size, **head_cfg_clean)
    
    # Set weight attribute for loss computation
    cls.weight = head_cfg.get("weight", 1.0)
    
    return cls


__all__ = [
    "OriginalHead",
    "PTMHead",
    "FunctionalRoleHead",
    "build_head",
    "HEAD_REGISTRY",
]

