"""
LoRA block: Low-Rank Adaptation block (Adapter-style with Layer Normalization).

Adapter-style LoRA formula:
△h = W_up(W_down(LN(h)))
h' = h + △h

This is more stable than standard LoRA due to Layer Normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init


class LoRABlock(nn.Module):
    """
    LoRA (Low-Rank Adaptation) block with Adapter-style.
    
    Implements Adapter-style LoRA adaptation:
    - Layer Normalization on input
    - W_down: (embed_dim, rank) - down projection
    - W_up: (rank, d_model) - up projection
    - Residual connection: h' = h + △h
    """
    
    def __init__(
        self,
        embed_dim: int,
        d_model: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        """
        Initialize LoRA block (Adapter-style with LN).
        
        @param embed_dim: Input embedding dimension
        @param d_model: Output model dimension
        @param rank: LoRA rank (low-rank dimension)
        @param alpha: LoRA scaling factor (typically alpha/rank is used)
        @param dropout: Dropout rate for LoRA adapters
        @param device: Device to place the model on
        @param dtype: Data type for the model
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Base linear layer (optional, can be identity if freeze_base is True)
        self.base_linear = nn.Linear(embed_dim, d_model, bias=False, **factory_kwargs)
        self.embed_dim = embed_dim
        self.d_model = d_model
        
        # LoRA parameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim, **factory_kwargs)
        
        # LoRA adapters: W_down (embed_dim, rank) and W_up (rank, d_model)
        # W_down: down projection
        self.lora_down = nn.Parameter(torch.empty(embed_dim, rank, **factory_kwargs))
        # W_up: up projection
        self.lora_up = nn.Parameter(torch.empty(rank, d_model, **factory_kwargs))
        
        # Initialize LoRA parameters
        # W_down: Kaiming uniform initialization
        init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        # W_up: Zero initialization (standard for adapters)
        init.zeros_(self.lora_up)
        
        # Dropout for LoRA adapters
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through LoRA block (Adapter-style).
        
        Formula:
        △h = W_up(W_down(LN(h)))
        h' = h + △h
        
        where:
        - h: input x
        - LN(h): Layer Normalization of h
        - W_down: down projection (embed_dim, rank)
        - W_up: up projection (rank, d_model)
        - △h: LoRA adaptation
        - h': output
        
        @param x: Input tensor of shape (batch_size, seq_len, embed_dim)
        @param **kwargs: Additional arguments (unused)
        @returns: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Base linear projection
        base_output = self.base_linear(x)  # (batch_size, seq_len, d_model)
        
        # Adapter-style LoRA: △h = W_up(W_down(LN(h)))
        # Step 1: Layer Normalization
        h_norm = self.layer_norm(x)  # (batch_size, seq_len, embed_dim)
        
        # Step 2: Down projection: W_down(LN(h))
        # h_norm: (batch_size, seq_len, embed_dim)
        # lora_down: (embed_dim, rank)
        # h_norm @ lora_down: (batch_size, seq_len, rank)
        h_down = self.dropout(h_norm) @ self.lora_down
        
        # Step 3: Up projection: W_up(h_down)
        # h_down: (batch_size, seq_len, rank)
        # lora_up: (rank, d_model)
        # h_down @ lora_up: (batch_size, seq_len, d_model)
        delta_h = h_down @ self.lora_up  # △h
        # Step 4: Residual connection: h' = h + △h
        # Scale delta_h by alpha/rank
        output = base_output + self.scaling * delta_h
        
        return output
