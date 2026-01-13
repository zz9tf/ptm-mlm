import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3, max_length=2000):
        super(TransformerClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * max_length, max_length),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max_length, 1)
        )

    def forward(self, binder, wt, ptm):
        binder_wt = torch.cat([binder, wt], dim=-1)
        binder_ptm = torch.cat([binder, ptm], dim=-1)
        x = self.fc(binder_wt - binder_ptm)
        return x.squeeze(-1)