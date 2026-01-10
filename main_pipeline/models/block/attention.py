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


class LoRABCBlock(nn.Module):
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
        d_attn: int = 64,
        num_attn_heads: int = 2,
        causal_attn: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Initialize LoRA block (Adapter-style with LN and tiny self-attention).

        @param embed_dim: Input embedding dimension
        @param d_model: Output model dimension
        @param rank: LoRA rank (low-rank dimension)
        @param alpha: LoRA scaling factor (typically alpha/rank is used)
        @param dropout: Dropout rate for LoRA adapters
        @param d_attn: Attention dimension for token communication (default: 64)
        @param num_attn_heads: Number of attention heads (default: 2)
        @param causal_attn: Whether to use causal attention (default: False)
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

        # Tiny self-attention for token communication
        self.d_attn = d_attn
        self.num_attn_heads = num_attn_heads
        self.head_dim = d_attn // num_attn_heads
        self.causal_attn = causal_attn

        # Attention projection layers
        self.attn_qkv = nn.Linear(embed_dim, 3 * d_attn, bias=False, **factory_kwargs)
        self.attn_out = nn.Linear(d_attn, embed_dim, bias=False, **factory_kwargs)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize attention parameters
        init.xavier_uniform_(self.attn_qkv.weight)
        init.xavier_uniform_(self.attn_out.weight)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through LoRA block (Adapter-style with tiny self-attention).

        Formula:
        △h = W_up(W_down(LN(h + Attn(LN(h)))))
        h' = h + △h

        where:
        - h: input x
        - LN(h): Layer Normalization of h
        - Attn(): Tiny self-attention for token communication
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

        # Adapter-style LoRA with attention: △h = W_up(W_down(LN(h + Attn(LN(h)))))
        # Step 1: Layer Normalization
        h_norm = self.layer_norm(x)  # (batch_size, seq_len, embed_dim)

        # Step 2: Tiny self-attention for token communication
        # QKV projection: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, 3 * d_attn)
        qkv = self.attn_qkv(h_norm)
        batch_size, seq_len, _ = qkv.shape

        # Split into Q, K, V: each (batch_size, seq_len, d_attn)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_attn) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)

        # Transpose for attention: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention computation
        # attn_weights: (batch_size, num_heads, seq_len, seq_len)
        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask if specified
        if self.causal_attn:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention: (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_weights @ v

        # Reshape back: (batch_size, seq_len, d_attn)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_attn)

        # Output projection back to embed_dim: (batch_size, seq_len, embed_dim)
        attn_out = self.attn_out(attn_output)

        # Residual connection for attention
        h_with_attn = h_norm + attn_out

        # Step 3: Down projection: W_down(h_with_attn)
        # h_with_attn: (batch_size, seq_len, embed_dim)
        # lora_down: (embed_dim, rank)
        # h_with_attn @ lora_down: (batch_size, seq_len, rank)
        h_down = self.dropout(h_with_attn) @ self.lora_down

        # Step 4: Up projection: W_up(h_down)
        # h_down: (batch_size, seq_len, rank)
        # lora_up: (rank, d_model)
        # h_down @ lora_up: (batch_size, seq_len, d_model)
        delta_h = h_down @ self.lora_up  # △h

        # Step 5: Residual connection: h' = h + △h
        # Scale delta_h by alpha/rank
        output = base_output + self.scaling * delta_h

        return output
