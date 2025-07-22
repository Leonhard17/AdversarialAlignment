import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.utils as pyg_utils
"""
Define the GNN model for aggregating attention graphs

It extracts information through message passing and aggregation
which is passed to the rest of the model.
This is mostly used to extract information and reduce to linear dimensionality
"""
class AggregationNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, embedding_dim=2, dropout=0.2, adj_dropout=0.2): # TODO: hyperparameter tuning for dropout
        super(AggregationNetwork, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim) # each node has a single feature (weight)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.adj_dropout = adj_dropout
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr  
        edge_index, edge_mask = pyg_utils.dropout_edge(edge_index, p=self.adj_dropout, training=self.training)

        if edge_weight is not None:
            edge_weight = edge_weight[edge_mask]

        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_mean_pool(x, data.batch)
        return x