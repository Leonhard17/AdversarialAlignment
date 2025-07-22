# test_alignment.py
import pickle
from pathlib import Path

import torch

# import your loader factory
from AttentionDataset.AttentionDataset import create_attention_loader

# import your model
from Model.full_model import FullAdversarialAlignmentModel

def main():
    BASE_DIR = Path(__file__).resolve().parent

    # 1) load the pickle
    pkl_path = BASE_DIR / "AttentionDataset" / "data" / "attention_dataset.pkl"
    with open(pkl_path, "rb") as f:
        attentions, rewards = pickle.load(f)

    # 2) build a small loader
    loader = create_attention_loader(attentions, rewards, batch_size=4)

    # 3) instantiate the model
    model = FullAdversarialAlignmentModel(
        num_iterations=5,
        num_layers=12,
        num_heads=12,
        gnn_hidden_dim=16,
        gnn_embedding_dim=8,
        compression_hidden_dim=32,
        compression_dim=4,
        agg_hidden_dim=4,
        agg_heads=2,
        agg_layers=2,
        reward_hidden_dim=24,
        reward_heads=2,
        reward_layers=2,
        reward_ff_dim=64,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) run a single batch through it
    attn_batch, reward_batch = next(iter(loader))
    # Move rewards to device
    reward_batch = reward_batch.to(device)

    # move each PyG Batch in the nested list to device
    for iteration_layers in attn_batch:
        for layer_heads in iteration_layers:
            for idx, graph in enumerate(layer_heads):
                layer_heads[idx] = graph.to(device)

    # forward
    preds = model(attn_batch).squeeze()  # shape (batch_size,)
    print("Predictions:", preds)
    print("Targets:    ", reward_batch)

if __name__ == "__main__":
    main()
