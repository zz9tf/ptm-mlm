"""
Functional Role head: Predicts functional role at PTM positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class FunctionalRoleHead(nn.Module):
    """
    Head for predicting functional role at PTM positions.
    Uses BCE loss for binary classification.
    """
    
    def __init__(self, d_model: int, output_size: int = 1, device=None, dtype=None, **kwargs):
        """
        Initialize functional role head.

        @param d_model: Model dimension for input features
        @param output_size: Output dimension (default: 1 for binary classification)
        @param device: Device to place the model on
        @param dtype: Data type for the model
        @param **kwargs: Additional arguments (unused)
        """
        super().__init__()
        self.type = "functional_role"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.output_size = output_size
        self.mix_type = kwargs.get("mix_type", "concat")
        if self.mix_type == "concat":
            self.head = nn.Linear(d_model*2, output_size, bias=False, **factory_kwargs)
        elif self.mix_type == "gate":
            self.gate_layer = nn.Linear(d_model*2, 1, bias=False, **factory_kwargs)
            self.head = nn.Linear(d_model, output_size, bias=False, **factory_kwargs)
        else:
            raise ValueError(f"Invalid mix_type: {self.mix_type}")
    
    def forward(self, features: torch.Tensor, processed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Predict functional role at PTM positions.

        @param features: Processed features of shape (batch_size, seq_len, d_model)
        @param processed: Dictionary of processed features from previous heads
        @param **kwargs: Additional arguments, must include ptm_position
        @returns: Logits of shape (batch_size, seq_len, output_size)
        """
        processed_features = processed.get("ptm_features", None)
        if processed_features is None:
            raise ValueError("ptm_features must be provided")

        # Get ptm_position from kwargs
        ptm_position = kwargs.get("ptm_position", None)
        if ptm_position is None:
            raise ValueError("ptm_position must be provided")
        if self.mix_type == "concat":
            features = torch.cat([features[:, ptm_position, :], processed_features[:, ptm_position, :]], dim=-1)
        elif self.mix_type == "gate":
            gate = torch.sigmoid(self.gate_layer(torch.cat([features[:, ptm_position, :], processed_features[:, ptm_position, :]], dim=-1)))
            features = features[:, ptm_position, :] * gate + processed_features[:, ptm_position, :] * (1 - gate)

        return {
            "logits": self.head(features)
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute BCE loss for functional role prediction.

        @param logits: Head output logits of shape (batch_size, seq_len, output_size)
        @param targets: Target values of shape (batch_size, seq_len) - binary labels
        @param device: Device for tensor operations
        @param **kwargs: Additional arguments, may include functional_role_mask
        @returns: Scalar loss tensor
        """
        # Get mask for positions where functional role should be predicted
        functional_role_mask = kwargs.get("functional_role_mask", None)
        if functional_role_mask is None:
            # Default: predict at all positions
            functional_role_mask = torch.ones_like(targets, dtype=torch.bool)

        # Only compute loss at masked positions
        if functional_role_mask.any():
            # For binary classification, squeeze the last dimension if output_size == 1
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return F.binary_cross_entropy_with_logits(
                logits[functional_role_mask],
                targets[functional_role_mask].float()
            )
        else:
            return torch.tensor(0.0, device=device)

