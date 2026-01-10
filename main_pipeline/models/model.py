"""
Main model class that combines blocks and heads.

Simple and direct: model(inputs) -> outputs
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from models.block import build_block
from models.head import build_head


class PTMModel(nn.Module):
    """
    Main model that combines blocks and heads.
    
    Structure:
    1. Block: Processes embeddings (e.g., LoRA, Mamba, etc.)
    2. Heads: Multiple heads that generate sequences
    
    Usage:
        model = PTMModel(embed_dim=5120, vocab_size=50277, d_model=512)
        outputs = model(input_ids, embeddings=embeddings)
    """
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        d_model: int,
        block_config: Optional[Dict[str, Any]] = None,
        heads_config: Optional[List[Dict[str, Any]]] = None,
        device=None,
        dtype=None,
    ):
        """
        Initialize PTM model.
        
        @param block_config: Configuration for block (e.g., {"type": "linear"})
        @param heads_config: Configuration for heads (e.g., [{"type": "original", "name": "original_sequence"}, {"type": "ptm", "name": "ptm_sequence"}])
        @param embed_dim: Dimension of input embeddings (ESM embedding dimension)
        @param d_model: Model dimension for intermediate processing
        @param vocab_size: Vocabulary size for output sequences
        @param device: Device to place the model on
        @param dtype: Data type for the model
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Step 1: Initialize block (processes embeddings)
        block_config_full = {
            "embed_dim": embed_dim,
            "d_model": d_model,
            **block_config,
            **factory_kwargs,
        }
        self.block = build_block(block_config_full.get("type", "lora"), block_config_full)
        
        # Step 2: Initialize heads (generate sequences)
        self.heads = nn.ModuleDict()
        if heads_config is None:
            # Default: original head and PTM head
            heads_config = [
                {"type": "original"},
                {"type": "ptm"},
            ]
        for head_cfg in heads_config:
            head_type = head_cfg['type']  # Use type as key in model.heads
            head_cfg.update(factory_kwargs)
            self.heads[head_type] = build_head(
                head_cfg=head_cfg,
                d_model=d_model,
                vocab_size=vocab_size
            )
        self.head_names = [head_cfg['type'] for head_cfg in heads_config]
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        @param input_ids: Input token IDs of shape (batch_size, seq_len), optional
        @param embeddings: ESM embeddings of shape (batch_size, seq_len, embed_dim), required
        @param **kwargs: Additional arguments passed to block and heads
        @returns: Dictionary mapping head names to their logits
            Each value is of shape (batch_size, seq_len, vocab_size)
        """
        if embeddings is None:
            raise ValueError("embeddings must be provided")

        # Check input embeddings for NaN/Inf before processing
        def _check0(name, t):
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                raise RuntimeError(f"{name} NaN/Inf, idx={bad.tolist()}")

        _check0("model_input_embeddings", embeddings)

        # Step 1: Process embeddings through block
        processed_features = self.block(embeddings, **kwargs)  # (batch_size, seq_len, d_model)
        
        # Step 2: Generate sequences from all heads
        results = {}
        processed = {}
        for head_name in self.head_names:
            head = self.heads[head_name]
            result = head(processed_features, processed, **kwargs)  # (batch_size, seq_len, vocab_size)
            results[head_name] = result['logits']
            for key, value in result.items():
                if key != 'logits':
                    processed[f"{head_name}_{key}"] = value
        return results
    
    def compute_loss(
        self,
        losses_compute_related: Dict[str, Any],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the model.

        @param losses_compute_related: Dictionary mapping head names to loss configurations, each containing:
            - logits: Logits to compute loss for
            - target: Target to compute loss for
            - kwargs: Additional arguments for loss function
        @param step: Current training step for loss weight warm-up
        @returns: Dictionary mapping loss names to their values, plus 'total' key
        """

        # Functional role loss weight warm-up function
        def fr_weight(step, w_max=0.1, t_warm=2000):
            """
            Linear warm-up for functional role loss weight.

            @param step: Current training step
            @param w_max: Maximum weight (final weight)
            @param t_warm: Warm-up steps
            @returns: Current weight
            """
            if step is None:
                return w_max  # If no step provided, use max weight
            return w_max * min(1.0, step / t_warm)

        losses = {'total': 0.0}
        for head_name in self.head_names:
            head = self.heads[head_name]
            loss_compute_related = losses_compute_related[head_name]

            # Extract logits and target from loss_compute_related
            logits = loss_compute_related['logits']
            target = loss_compute_related['target']  # This will be passed as input_ids

            # Call head.compute_loss with target as input_ids
            loss_value = head.compute_loss(logits, target, **loss_compute_related['kwargs'])
            losses[head_name] = loss_value

            # Apply warm-up for functional_role head
            if head_name == 'functional_role':
                base_weight = getattr(head, 'weight', 0.1)
                head_weight = fr_weight(step, w_max=base_weight, t_warm=3000)
                # Log weight progression every 100 steps
                if step is not None and step % 100 == 0:
                    print(f"Functional role weight at step {step}: {head_weight:.6f} (target: {base_weight:.6f})")
            else:
                # Use head.weight if available, otherwise default to 1.0
                head_weight = getattr(head, 'weight', 1.0)

            losses['total'] += loss_value * head_weight

        return losses
