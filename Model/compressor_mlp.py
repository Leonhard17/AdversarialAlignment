import torch
import torch.nn as nn

"""
Define the Compression Network to compress data from the AggregationNetwork

Compresses the embedding from the AggregationNetworks (Heads, Layers)
into a smaller representation. This is then passed to the adversarial transformer model.
"""
class CompressionNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_dim, dropout=0.1):
        super(CompressionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, compressed_dim)
        )

    def forward(self, x):
        x = self.net(x) 
        return x