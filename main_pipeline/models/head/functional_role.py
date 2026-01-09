"""
Functional Role head: Predicts functional role at PTM positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class FunctionalRoleHead(nn.Module):
    """
    Head for predicting functional role at PTM positions in protein sequences.
    Uses BCE loss for binary classification at labeled positions only.
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
    
    def forward(self, features: torch.Tensor, processed: Dict[str, Any], position_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict functional role using pooled features from PTM positions.

        @param features: Processed features of shape (batch_size, seq_len, d_model)
        @param processed: Dictionary of processed features from previous heads
        @param position_mask: Boolean mask indicating PTM positions (batch_size, seq_len)
        @param **kwargs: Additional arguments
        @returns: Dictionary containing logits of shape (batch_size, output_size)
        """
        processed_features = processed.get("ptm_features", None)
        if processed_features is None:
            raise ValueError("ptm_features must be provided")

        # Extract features at PTM positions
        masked_features = features[position_mask]  # (num_masked_positions, d_model)
        masked_ptm_features = processed_features[position_mask]  # (num_masked_positions, d_model)

        if self.mix_type == "concat":
            combined_features = torch.cat([masked_features, masked_ptm_features], dim=-1)
        elif self.mix_type == "gate":
            gate_input = torch.cat([masked_features, masked_ptm_features], dim=-1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            combined_features = masked_features * gate + masked_ptm_features * (1 - gate)

        # Compute logits for masked positions
        logits = self.head(combined_features)  # (num_masked_positions, output_size)

        return {
            "logits": logits
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

        @param logits: Head output logits of shape (batch_size, 1) - already masked in forward
        @param targets: Target values of shape (batch_size, 1) - binary labels (0/1)
        @param device: Device for tensor operations
        @param **kwargs: Additional arguments
        @returns: Scalar loss tensor
        """
        if targets.numel() == 0 or logits.numel() == 0:
            return torch.tensor(0.0, device=device)

        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),  # (batch_size,)
            targets.squeeze(-1).float()  # (batch_size,)
        )

