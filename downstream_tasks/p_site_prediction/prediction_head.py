"""
Classification head for phosphorylation site prediction.
This module provides a classification head that can be trained on top of pre-trained embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for per-position site prediction.
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