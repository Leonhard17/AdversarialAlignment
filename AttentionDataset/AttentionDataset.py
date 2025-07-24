from pathlib import Path
from typing import Any, List, Tuple

import networkx as nx
import torch
from typing import Callable, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

# NOTE: The attention is preprocessed so this function is not used in the DatasetGenerator anymore
# TODO: Pre do this for all heads and layers, then save as a single file
# TODO: Collate to device
# def attention_collate_fn(
#     batch: List[Tuple[List[List[torch.Tensor]], float]]
# ) -> Tuple[List[List[List[List[Batch]]]], torch.Tensor]:
#     """
#     Collate a batch of (attentions, reward) pairs into:
#       - List of samples → List of steps → List of layers → List of head PyG Data objects
#       - Tensor of rewards
#     """
#     attentions_list, rewards_list = zip(*batch)
#     rewards = torch.tensor(rewards_list, dtype=torch.float)

#     collated = []
#     for sample_attns in attentions_list:
#         sample_steps: List[List[List[Batch]]] = []
#         for step_attns in sample_attns:
#             layer_list: List[List[Batch]] = []
#             for layer_attn in step_attns:
#                 # layer_attn: (1, num_heads, seq_len, seq_len)
#                 heads = layer_attn.squeeze(0)
#                 head_list: List[Batch] = []
#                 for head_mat in heads:
#                     G = attention_to_graph(head_mat)
#                     pyg_data = from_networkx(G, group_node_attrs=["weight"], group_edge_attrs=["weight"])
#                     head_list.append(pyg_data)
#                 layer_list.append(Batch.from_data_list(head_list))
#             sample_steps.append(layer_list)
#         collated.append(sample_steps)

#     return collated, rewards


def attention_collate_fn(
    device: torch.device
) -> Callable[
       [List[Tuple[List[List[torch.Tensor]], float]]],
       Tuple[List[List[List[List[Batch]]]], torch.Tensor]
   ]:
    """
    Returns collate_fn which:
    Collates a batch of (attentions, reward) pairs into and puts them on device:
      - List of samples → List of steps → List of layers → List of head PyG Data objects
      - Tensor of rewards
    """
    def collate_move(
        batch: List[Tuple[List[List[torch.Tensor]], float]]
    ) -> Tuple[List[List[List[List[Batch]]]], torch.Tensor]:
        # extract batch elements
        attentions_list, rewards_list = zip(*batch)
        # convert rewards to tensor and move to device
        rewards = torch.tensor(rewards_list, dtype=torch.float).to(device, non_blocking=True).float()
        # move attention graphs to device
        for it in attentions_list:
                for layer_heads in it:
                    for i, g in enumerate(layer_heads):
                        layer_heads[i] = g.to(device)
        return attentions_list, rewards
    
    return collate_move


class AttentionDataset(Dataset):
    """
    PyTorch Dataset for attention weights + scalar rewards.
    """
    def __init__(self, attentions: List[Any], rewards: List[float]):
        self.attentions = attentions
        self.rewards = rewards

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx: int) -> Tuple[List[List[torch.Tensor]], float]:
        return self.attentions[idx], self.rewards[idx]


def create_attention_loader(
    attentions: List[Any],
    rewards: List[float],
    batch_size: int = 32,
    shuffle: bool = False,
    device: torch.device = torch.device("cpu")
) -> DataLoader:
    """
    Build a DataLoader over AttentionDataset using our custom collate_fn.
    """
    dataset = AttentionDataset(attentions, rewards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=attention_collate_fn(device), #NOTE: Data now preprocessed
        # TODO: make features below adjustable
        num_workers=0, # Use 0 for Windows compatibility
        pin_memory=False,
        # persistent_workers=True, windows issues
        #prefetch_factor=2,
    )


if __name__ == "__main__":
    # Example: load precomputed attention_data & reward_data, then iterate one batch
    BASE_DIR = Path(__file__).resolve().parent
    pkl_path = BASE_DIR / "data" / "attention_dataset.pkl"

    # Attentions: List of samples → List of steps → List of layer‑tensors
    # Rewards:    List of floats
    attentions, rewards = torch.load(pkl_path) if pkl_path.suffix == ".pt" else __import__("pickle").load(pkl_path.open("rb"))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_attention_loader(attentions, rewards, batch_size=32, shuffle=False, device=device)
    for batch_graphs, batch_rewards in loader:
        print(f"Batch size = {len(batch_graphs)}, Rewards = {batch_rewards}")
        # batch_graphs is List[ steps ] where each step is a PyG Batch of all heads/layers
        break
