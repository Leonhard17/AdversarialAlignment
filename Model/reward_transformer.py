import torch
import torch.nn as nn

""" 
Define the AttentionToRewardEncoder used to predict rewards

This encoder processes the attention features and predicts a reward based on them.
Used to predict the reward over multiple iterations of the primary Network.
"""
class AttentionToRewardEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_head=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(AttentionToRewardEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)  # Project attention features
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head , 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(hidden_dim, 1)  # Predict reward

        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_dim))  # Positional encoding

    def forward(self, x):
        x = self.embedding(x)  
        x = self.dropout(x)
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional info
        x = self.transformer_encoder(x)  # No causal mask needed
        x = x.mean(dim=1)  # Pool over token representations (global understanding)
        x = self.fc_out(x)  # Predict token logits  
        
        return x  # Output shape: (reward)