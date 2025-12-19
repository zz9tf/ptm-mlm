import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class EmbeddingMLP(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, lr=1e-3):
        super(EmbeddingMLP, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
        embedding_dim=embedding_dim)
        self.fc = nn.Sequential(
        nn.Linear(embedding_dim, 2)
        )
        self.lr = lr
    def forward(self, x):
        x = x.to(torch.int32)
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        linear_output = self.fc(embedded)
        softmax_output = F.log_softmax(linear_output, dim=1)
        return softmax_output