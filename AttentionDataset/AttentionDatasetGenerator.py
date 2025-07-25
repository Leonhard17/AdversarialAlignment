from pathlib import Path
import re
import pickle
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from typing import List, Tuple, Any
from tqdm.auto import tqdm

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

def load_math_data(
    problems_path: Path,
    solutions_path: Path
) -> pd.DataFrame:
    """
    Load math problems and solutions into a DataFrame.
    """
    problems = problems_path.read_text().splitlines()
    solutions = solutions_path.read_text().splitlines()
    return pd.DataFrame({"problem": problems, "solution": solutions})


def setup_model(
    checkpoint_dir: Path,
    base_model: str = "gpt2"
) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel, torch.device]:
    if checkpoint_dir.is_dir():
        source = str(checkpoint_dir)
    else:
        print(f"Warning! '{checkpoint_dir}' not found — loading '{base_model}' instead.")
        source = base_model
    """
    Load tokenizer and model from a fine-tuned checkpoint directory.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.pad_token = '<|pad|>'
    model = GPT2LMHeadModel.from_pretrained(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def generate_with_attentions(
    problem: str,
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    device: torch.device,
    max_new_tokens: int = 10
) -> Tuple[str, List[List[torch.Tensor]]]:
    """
    Generate text and per-step attentions for a single problem.
    Returns the decoded text and a list of attention tensors.
    """
    model.eval()
    attentions_per_step: List[List[torch.Tensor]] = []
    enc = tokenizer(problem, return_tensors="pt")
    cur_ids = enc.input_ids.to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=cur_ids,
                attention_mask=torch.ones_like(cur_ids),
                output_attentions=True,
                return_dict=True
            )
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            attentions_per_step.append([a.cpu() for a in outputs.attentions])
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    full_seq = cur_ids[0].cpu().tolist()
    text = tokenizer.decode(full_seq, skip_special_tokens=True)
    return text, attentions_per_step


def trim_solution(solution_text: str, split_token: str = " =") -> str:
    """
    Remove everything before the split_token and return only the first line.
    """
    idx = solution_text.find(split_token)
    trimmed = solution_text[idx + len(split_token):].strip() if idx != -1 else solution_text.strip()
    return trimmed.splitlines()[0]


def extract_number(text: str) -> str:
    """
    Extract the first floating-point number from text using regex.
    """
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?" # Regex for floating-point numbers
    match = re.search(pattern, text)
    if not match:
         raise ValueError(f"No numeric value found in '{text}'")
    return match.group(0)


def build_attention_dataset(
    data: pd.DataFrame,
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    device: torch.device,
    num_samples: int = 100,
    max_new_tokens: int = 10
) -> Tuple[List[Any], List[float]]:
    """
    Generate attentions and rewards for the first num_samples problems.
    """
    attention_data: List[Any] = []
    reward_data: List[float] = []

    for idx in tqdm(range(min(num_samples, len(data))), desc="Generating attentions"):
        problem = data.iloc[idx, 0]
        true_solution = data.iloc[idx, 1]
        raw_text, attentions = generate_with_attentions(
            problem, tokenizer, model, device, max_new_tokens
        )
        ans_line = trim_solution(raw_text)
        try:
            num_str = extract_number(ans_line)
            diff = abs(float(num_str) - float(true_solution)) + 1e-6
            reward = -torch.log(torch.tensor(diff)).item()

        except ValueError:
            print(f"Skipping sample {idx}: No numeric value found in '{ans_line}'")
            continue

        # Convert once to torch_geometric.data.Data

        sample_graphs = []
        for step_attns in attentions:
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
            sample_graphs.append(layer_list)
        attention_data.append(sample_graphs)
        reward_data.append(reward)

    return attention_data, reward_data



def save_attention_dataset(
    attention_data: List[Any],
    reward_data: List[float],
    out_path: Path
) -> None:
    """
    Save attention_data and reward_data to a pickle file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump((attention_data, reward_data), f)


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / "MathDataset" / "data"
    issues = load_math_data(
        problems_path=data_dir / "math_problems.txt",
        solutions_path=data_dir / "math_solutions.txt"
    )
    tokenizer, model, device = setup_model(
        checkpoint_dir=BASE_DIR / "finetuned_models" / "finetuned_multiplication_gpt2_epoch_5"
    )

    print("Generating attention dataset...")
    attention_data, reward_data = build_attention_dataset(
        data=issues,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_samples=4000,
        max_new_tokens=10
    )

    data_dir = BASE_DIR / f"data"
    data_dir.mkdir(exist_ok=True)
    out_path = BASE_DIR / "data" /"attention_dataset.pkl"
    save_attention_dataset(attention_data, reward_data, out_path)
    print(f"Dataset saved with {len(reward_data)} entries to {out_path}")


if __name__ == "__main__":
    main()
