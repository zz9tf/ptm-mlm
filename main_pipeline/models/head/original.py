"""
Original head: Generates original sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class OriginalHead(nn.Module):
    """
    Head for generating original sequence.
    Loss is computed at all non-padding positions.
    """
    
    def __init__(self, d_model: int, vocab_size: int, device=None, dtype=None, **kwargs):
        """
        Initialize original head.
        
        @param d_model: Model dimension for input features
        @param vocab_size: Vocabulary size for output sequences
        @param device: Device to place the model on
        @param dtype: Data type for the model
        @param **kwargs: Additional arguments (unused)
        """
        super().__init__()
        self.type = "original"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
    
    def forward(self, features: torch.Tensor, processed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Generate original sequence logits.

        @param features: Processed features of shape (batch_size, seq_len, d_model)
        @param processed: Dictionary of processed features from previous heads
        @param **kwargs: Additional arguments (unused)
        @returns: Logits of shape (batch_size, seq_len, vocab_size)
        """
        logits = self.head(features)

        return {
            "logits": logits
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
        # 🎯 使用 attention_mask 来过滤有效位置
        # attention_mask 已经根据 range 数据正确生成，包含了所有有效位置
        if attention_mask is not None:
            valid_mask = attention_mask  # attention_mask 已经正确标记了有效位置
        else:
            # Fallback: 如果没有 attention_mask，使用原来的逻辑
            valid_mask = input_ids != 0
        
        if valid_mask.any():
            return F.cross_entropy(
                logits[valid_mask],
                input_ids[valid_mask],
            )
        else:
            return torch.tensor(0.0, device=device)

