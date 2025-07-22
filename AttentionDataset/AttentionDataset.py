from pathlib import Path
from typing import Any, List, Tuple

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx


def attention_to_graph(attention: torch.Tensor) -> nx.DiGraph:
    """
    Convert a single-head attention matrix (seq_len × seq_len) into a directed graph.
    Nodes are token positions with a 'weight' = self‑attention score.
    Edges (i→j) exist for j < i, with 'weight' = attention[i, j].
    """
    seq_len = attention.size(-1)
    G = nx.DiGraph()
    # Add nodes
    for i in range(seq_len):
        G.add_node(i, weight=float(attention[i, i]))
    # Add masked edges
    for i in range(seq_len):
        for j in range(i):
            G.add_edge(i, j, weight=float(attention[i, j]))
    return G


def attention_collate_fn(
    batch: List[Tuple[List[List[torch.Tensor]], float]]
) -> Tuple[List[List[List[List[Batch]]]], torch.Tensor]:
    """
    Collate a batch of (attentions, reward) pairs into:
      - List of samples → List of steps → List of layers → List of head PyG Data objects
      - Tensor of rewards
    """
    attentions_list, rewards_list = zip(*batch)
    rewards = torch.tensor(rewards_list, dtype=torch.float)

    collated = []
    for sample_attns in attentions_list:
        sample_steps: List[List[List[Batch]]] = []
        for step_attns in sample_attns:
            layer_list: List[List[Batch]] = []
            for layer_attn in step_attns:
                # layer_attn: (1, num_heads, seq_len, seq_len)
                heads = layer_attn.squeeze(0)
                head_list: List[Batch] = []
                for head_mat in heads:
                    G = attention_to_graph(head_mat)
                    pyg_data = from_networkx(G, group_node_attrs=["weight"], group_edge_attrs=["weight"])
                    head_list.append(pyg_data)
                layer_list.append(Batch.from_data_list(head_list))
            sample_steps.append(layer_list)
        collated.append(sample_steps)

    return collated, rewards


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
    shuffle: bool = False
) -> DataLoader:
    """
    Build a DataLoader over AttentionDataset using our custom collate_fn.
    """
    dataset = AttentionDataset(attentions, rewards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=attention_collate_fn
    )


if __name__ == "__main__":
    # Example: load precomputed attention_data & reward_data, then iterate one batch
    BASE_DIR = Path(__file__).resolve().parent
    pkl_path = BASE_DIR / "data" / "attention_dataset.pkl"

    # Attentions: List of samples → List of steps → List of layer‑tensors
    # Rewards:    List of floats
    attentions, rewards = torch.load(pkl_path) if pkl_path.suffix == ".pt" else __import__("pickle").load(pkl_path.open("rb"))

    loader = create_attention_loader(attentions, rewards, batch_size=32, shuffle=False)
    for batch_graphs, batch_rewards in loader:
        print(f"Batch size = {len(batch_graphs)}, Rewards = {batch_rewards}")
        # batch_graphs is List[ steps ] where each step is a PyG Batch of all heads/layers
        break
