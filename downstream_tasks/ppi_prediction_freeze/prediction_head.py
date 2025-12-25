"""
Classification head for protein-protein interaction (PPI) prediction.
This module provides a classification head that can be trained on top of pre-trained embeddings.
For PPI prediction, this is a sequence-level classification task (not per-position).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIClassificationHead(nn.Module):
    """
    Classification head for sequence-level PPI prediction.
    Takes sequence embeddings and outputs binary classification logits.
    """
    
    def __init__(self, hidden_size=128, num_labels=2, dropout=0.1, pooling_method='mean'):
        """
        Initialize the classification head.
        
        @param hidden_size: Size of input embeddings (default: 128)
        @param num_labels: Number of output classes (default: 2 for binary classification)
        @param dropout: Dropout rate (default: 0.1)
        @param pooling_method: Pooling method for sequence-level representation ('mean', 'max', 'cls', or 'attention')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling_method = pooling_method
        
        # Attention pooling layer (if using attention pooling)
        if pooling_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Classification layers
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features, attention_mask=None, **kwargs):
        """
        Forward pass through the classification head.
        
        @param features: Input embeddings tensor of shape (batch_size, seq_len, hidden_size)
        @param attention_mask: Attention mask tensor of shape (batch_size, seq_len) (optional)
        @param **kwargs: Additional keyword arguments (unused)
        @returns: Logits tensor of shape (batch_size, num_labels)
        """
        # Pool sequence embeddings to get sequence-level representation
        if self.pooling_method == 'mean':
            # Mean pooling (masked)
            if attention_mask is not None:
                # Mask out padding positions
                mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                masked_features = features * mask  # [batch_size, seq_len, hidden_size]
                seq_repr = masked_features.sum(dim=1) / mask.sum(dim=1)  # [batch_size, hidden_size]
            else:
                seq_repr = features.mean(dim=1)  # [batch_size, hidden_size]
        
        elif self.pooling_method == 'max':
            # Max pooling (masked)
            if attention_mask is not None:
                # Set padding positions to very negative values
                mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                masked_features = features * mask + (1 - mask) * (-1e9)
                seq_repr = masked_features.max(dim=1)[0]  # [batch_size, hidden_size]
            else:
                seq_repr = features.max(dim=1)[0]  # [batch_size, hidden_size]
        
        elif self.pooling_method == 'cls':
            # Use first token (CLS token) as sequence representation
            seq_repr = features[:, 0, :]  # [batch_size, hidden_size]
        
        elif self.pooling_method == 'attention':
            # Attention pooling
            if attention_mask is not None:
                # Compute attention weights
                attn_weights = self.attention(features)  # [batch_size, seq_len, 1]
                # Mask out padding positions
                mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                attn_weights = attn_weights * mask + (1 - mask) * (-1e9)
                attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
                # Weighted sum
                seq_repr = (features * attn_weights).sum(dim=1)  # [batch_size, hidden_size]
            else:
                attn_weights = self.attention(features)  # [batch_size, seq_len, 1]
                attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
                seq_repr = (features * attn_weights).sum(dim=1)  # [batch_size, hidden_size]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Classification layers
        x = self.dropout(seq_repr)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x


class ClassificationHead(nn.Module):
    """
    Classification head for per-position site prediction (for compatibility with p_site_prediction).
    Takes embeddings and outputs binary classification logits for each position.
    """
    
    def __init__(self, hidden_size=128, num_labels=2, dropout=0.1):
        """
        Initialize the classification head.
        
        @param hidden_size: Size of input embeddings (default: 128)
        @param num_labels: Number of output classes (default: 2 for binary classification)
        @param dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features, **kwargs):
        """
        Forward pass through the classification head.
        
        @param features: Input embeddings tensor of shape (batch_size, seq_len, hidden_size) 
                        or (batch_size, hidden_size)
        @param **kwargs: Additional keyword arguments (unused)
        @returns: Logits tensor of shape (batch_size, seq_len, num_labels) or (batch_size, num_labels)
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x














