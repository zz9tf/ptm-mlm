"""
PTM head: Generates PTM sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


class PTMHead(nn.Module):
    """
    Head for generating PTM sequence.
    Loss is computed at PTM site positions only.
    """
    
    def __init__(self, d_model: int, vocab_size: int, device=None, dtype=None):
        """
        Initialize PTM head.
        
        @param d_model: Model dimension for input features
        @param vocab_size: Vocabulary size for output sequences
        @param device: Device to place the model on
        @param dtype: Data type for the model
        """
        super().__init__()
        self.type = "ptm"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.middle_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

    def forward(self, features: torch.Tensor, processed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Generate PTM sequence logits.

        @param features: Processed features of shape (batch_size, seq_len, d_model)
        @param processed: Dictionary of processed features from previous heads
        @param **kwargs: Additional arguments (unused)
        @returns: Logits of shape (batch_size, seq_len, vocab_size)
        """
        features = self.middle_layer(features)
        logits = self.head(features)

        return {
            "logits": logits,
            "features": features
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        device: torch.device,
        attention_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss at positions where attention_mask is True.

        @param logits: Head output logits of shape (batch_size, seq_len, vocab_size)
        @param input_ids: Target token IDs of shape (batch_size, seq_len)
        @param device: Device for tensor operations
        @param attention_mask: Attention mask of shape (batch_size, seq_len), where True indicates valid positions
        @param **kwargs: Additional arguments (unused)
        @returns: Scalar loss tensor
        """
        # 🎯 使用 attention_mask 来过滤有效位置（如果提供）
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len), True 表示有效位置
            valid_mask = attention_mask.view(-1)  # Flatten to (batch_size * seq_len,)
            logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            input_ids_flat = input_ids.view(-1)  # (batch_size * seq_len,)
            
            if valid_mask.any():
                return F.cross_entropy(
                    logits_flat[valid_mask],
                    input_ids_flat[valid_mask],
                    reduction='mean'
                )
            else:
                return torch.tensor(0.0, device=device)
        else:
            # Fallback: 如果没有 attention_mask，使用原来的逻辑
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction='mean'
            )

