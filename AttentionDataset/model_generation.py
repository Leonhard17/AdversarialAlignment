"""
Programm to check the model output, for sanity checks.
"""

from pathlib import Path
import re
import pickle
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Tuple, Any
from tqdm.auto import tqdm

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

def generate_solution(
    problem: str,
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    device: torch.device,
    max_new_tokens: int = 10
) -> Tuple[str, List[List[torch.Tensor]]]:
    """
    Generate text for a single problem.
    Returns the decoded text and a list of attention tensors.
    """
    model.eval()
    input_ids = tokenizer(problem, return_tensors="pt").input_ids.to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=False,
                return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        
    full_text = tokenizer.decode(
        input_ids[0].cpu().tolist(),
        skip_special_tokens=True)
    return full_text

def main(
    problems_path: Path,
    solutions_path: Path,
    checkpoint_dir: Path,
    num_samples: int = 1000,
    max_new_tokens: int = 10
) -> Tuple[List[Any], List[float]]:
    """
    Main function to generate attentions and rewards.
    """
    data = load_math_data(problems_path, solutions_path)
    tokenizer, model, device = setup_model(checkpoint_dir)

    attention_data: List[Any] = []
    reward_data: List[float] = []

    for idx in range(min(num_samples, len(data))):
        problem = data.iloc[idx, 0]
        solution = data.iloc[idx, 1]
        raw_text = generate_solution(
            problem, tokenizer, model, device, max_new_tokens
        )
        
        print(f"Sample {idx}: Problem: {problem}, Solution: {solution}, Generated: {raw_text}")

    return attention_data, reward_data

if __name__ == "__main__":
    # Example usage
    BASE_DIR = Path(__file__).resolve().parent
    problems_path = BASE_DIR / "MathDataset" / "data" / "math_problems.txt"
    solutions_path = BASE_DIR / "MathDataset" / "data" / "math_solutions.txt"
    checkpoint_dir = BASE_DIR / "finetuned_models" / "finetuned_multiplication_gpt2_epoch_5"

    attentions, rewards = main(problems_path, solutions_path, checkpoint_dir, num_samples=100)
    print(f"Generated {len(attentions)} samples with rewards.")