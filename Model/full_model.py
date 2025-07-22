import torch
import torch.nn as nn

# from aggregation_gnn import AggregationNetwork
# from compressor_mlp import CompressionNetwork
# from aggregation_transformer import AggregationEncoderTransformer
# from reward_transformer import AttentionToRewardEncoder

from Model import (
    AggregationNetwork,
    CompressionNetwork,
    AggregationEncoderTransformer,
    AttentionToRewardEncoder
)

"""
Full model that combines all components for adversarial alignment
"""
class FullAdversarialAlignmentModel(nn.Module):
    def __init__(
        self,
        num_iterations,
        num_layers,
        num_heads,

        gnn_hidden_dim,
        gnn_embedding_dim,

        compression_hidden_dim,
        compression_dim,

        agg_hidden_dim,
        agg_heads,
        agg_layers,
        
        reward_hidden_dim,
        reward_heads,
        reward_layers,
        reward_ff_dim,
        dropout=0.1
    ):
        super(FullAdversarialAlignmentModel, self).__init__()

        # needed variables
        self.num_iterations = num_iterations # used for padding

        # Shared GNN across all heads, layers, iterations
        self.gnn = AggregationNetwork(
            hidden_dim=gnn_hidden_dim,
            embedding_dim=gnn_embedding_dim,
            dropout=dropout
        )

        # Separate compression for each layer (compress across heads)
        self.compressors = nn.ModuleList([
            CompressionNetwork(
                input_dim=gnn_embedding_dim * num_heads,
                hidden_dim=compression_hidden_dim,
                compressed_dim=compression_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Shared AggregationEncoder for linking compressed layers inside each iteration
        self.aggregation_encoder = AggregationEncoderTransformer(
            input_dim=compression_dim * num_layers,  # Concatenate all compressed layers
            output_dim=agg_hidden_dim,  # Output dimension for each iteration summary
            hidden_dim=agg_hidden_dim,
            num_heads=agg_heads,
            num_layers=agg_layers,
            dropout=dropout
        )

        # Final reward predictor processing the sequence of iteration summaries
        self.reward_predictor = AttentionToRewardEncoder(
            input_dim=agg_hidden_dim * num_iterations,  # Concatenate all iteration summaries
            hidden_dim=reward_hidden_dim,
            num_head=reward_heads,
            num_layers=reward_layers,
            dim_feedforward=reward_ff_dim,
            dropout=dropout
        )

    def forward(self, attention_graph_batches):
        """
        Parameters
        ----------
        attention_graph_batches : list of lists of lists
            [iteration][layer][head] -> each element is a torch_geometric Batch

        Returns
        -------
        Tensor
            [batch_size, 1] reward predictions
        """

        iteration_embeddings = []
        for iteration_layers in attention_graph_batches:
            layer_embeddings = []
            for layer_idx, layer_heads in enumerate(iteration_layers):
                
                head_embeddings = []
                for head_graph in layer_heads:
                    # Shared GNN across all
                    gnn_out = self.gnn(head_graph)  # [batch_size, gnn_embedding_dim]
                    head_embeddings.append(gnn_out)

                # Concat all heads for this layer
                head_concat = torch.cat(head_embeddings, dim=-1)  # [batch_size, gnn_embedding_dim * num_heads]
                # Compress layer representation with layer-specific compressor
                compressed = self.compressors[layer_idx](head_concat)  # [batch_size, compression_dim]
                layer_embeddings.append(compressed)  # keep sequence dimension

            # Sequence of compressed layers for this iteration
            layer_seq = torch.stack(layer_embeddings, dim=0)  # [batch_size, num_layers, compression_dim]
            layer_seq_flat = layer_seq.view(layer_seq.size(0), -1)  # [batch_size, num_layers * compression_dim]
            iter_encoded = self.aggregation_encoder(layer_seq_flat.unsqueeze(1))  # [batch_size, 1, agg_hidden_dim]

            # Pool over layers (mean) to get single iteration summary
            iter_summary = iter_encoded.mean(dim=1).unsqueeze(1)  # [batch_size, 1, agg_hidden_dim]
            iteration_embeddings.append(iter_summary)

        # Padding
        dummy = torch.zeros((1, 1, iteration_embeddings[0].size(2)), device=iteration_embeddings[0].device)
        for i in range(len(iteration_embeddings)):
            num_missing = self.num_iterations - iteration_embeddings[i].size(0)
            if num_missing > 0:
                # Append num_missing copies to the tensor
                pad = dummy.repeat(num_missing, 1, 1)
                iteration_embeddings[i] = torch.cat([pad, iteration_embeddings[i]], dim=0)
            

        
        # Stack all iteration summaries into a sequence
        iteration_seq = torch.cat(iteration_embeddings, dim=1)  # [batch_size, num_iterations, agg_hidden_dim]
        iteration_seq_flat = iteration_seq.view(iteration_seq.size(1), -1)  # [batch_size, num_iterations * agg_hidden_dim]
        # Final reward prediction
        reward = self.reward_predictor(iteration_seq_flat.unsqueeze(1))  # [batch_size, 1, input_dim]
        return reward
    

# Example usage:
if __name__ == "__main__":
    model = FullAdversarialAlignmentModel(
        num_iterations=5,
        num_layers=3,
        num_heads=4,
        gnn_hidden_dim=64,
        gnn_embedding_dim=32,
        compression_hidden_dim=128,
        compression_dim=64,
        agg_hidden_dim=256,
        agg_heads=8,
        agg_layers=2,
        reward_hidden_dim=512,
        reward_heads=8,
        reward_layers=6,
        reward_ff_dim=2048
    )
    print(model)