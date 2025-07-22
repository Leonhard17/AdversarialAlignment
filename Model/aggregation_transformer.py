import torch
import torch.nn as nn

""" 
Define the AggregationEncoderTransformer used to link the different layers

Encoder only transformer that processes the aggregated embeddings, to link them togheter
and further compress a signle network state.
"""
class AggregationEncoderTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(AggregationEncoderTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Output layer
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_dim))  # Positional encoding

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x