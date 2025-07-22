# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from pathlib import Path

# Define the MathDataset class
class MathDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        problem = self.dataframe.iloc[idx]["problem"]
        solution = self.dataframe.iloc[idx]["solution"]

        return problem, solution

    
def make_collate_fn(tokenizer):
    # collate_fn function to handle padding and tokenization for a whole batch
    def collate_fn(batch):
        problems, solutions = zip(*batch)
        split_token = " =" # has additional space in front as this is a special token
        split_token_id = tokenizer.encode(split_token)[0]

        questions = [f"{p} {s}{tokenizer.eos_token}" for p, s in zip(problems, solutions)] # concatenate and add eos_token

        encoder = tokenizer(
            questions,  # Concatenate problems and solutions for encoding
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=20, # TODO: Adjust max_length based on model
            return_tensors="pt"
        )

        # mask the labels for the solutions
        labels = encoder["input_ids"].clone()
        for i in range(len(problems)):
            # Find the index of the split token in the input_ids
            split_index = (encoder["input_ids"][i] == split_token_id).nonzero(as_tuple=True)[0]
            # Set the labels to -100 for the problem part, so they won't be used in loss calculation
            labels[i][:(split_index+1)] = -100

        return {
            "input_ids": encoder["input_ids"],
            "attention_mask": encoder["attention_mask"],
            "labels": labels,  # Use the masked labels for loss calculation
        }
    return collate_fn

# Load the math dataset
def load_math_data(problem_filename="math_problems.txt", solution_filename="math_solutions.txt"):
    import pandas as pd
    problems = [line.strip() for line in open(problem_filename, "r")]
    solutions = [line.strip() for line in open(solution_filename, "r")]
    return pd.DataFrame({"problem": problems, "solution": solutions})

# Funtion to create DataLoader for training and testing
def create_data_loaders(problem_filename="data/math_problems.txt", solution_filename="data/math_solutions.txt", 
                        batch_size=32, tokenizer_name="gpt2"):
    """
    Create DataLoaders for training and testing the MathDataset.
    Args:
        problem_filename (str): Path to the file containing math problems.
        solution_filename (str): Path to the file containing solutions.
        batch_size (int): Batch size for DataLoader.
        tokenizer_name (str): Name of the tokenizer to use.
    Returns:
        train_data_loader (DataLoader): DataLoader for training data.
        test_data_loader (DataLoader): DataLoader for testing data.
        tokenizer (GPT2Tokenizer): Tokenizer used for encoding.
    """

    data = load_math_data(problem_filename, solution_filename)
    
    # split into train and test sets
    # NOTE: This is a simple split, consider using sklearn's train_test_split for more complex scenarios
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Initialize tokenizer and dataset
    # NOTE: Change depedning on the model you are using
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})  # Explicitly add a special padding token
        tokenizer.pad_token = '<|pad|>'

    train_math_dataset = MathDataset(train_data)
    test_math_dataset = MathDataset(test_data)

    train_data_loader = DataLoader(train_math_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(tokenizer))
    test_data_loader = DataLoader(test_math_dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))

    return train_data_loader, test_data_loader, tokenizer

# Test the dataset loading
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    problem_file = BASE_DIR / "data" / "math_problems.txt"
    solution_file = BASE_DIR / "data" / "math_solutions.txt"

    train_loader, test_loader, tokenizer = create_data_loaders(problem_filename=problem_file, solution_filename=solution_file, batch_size=10)
    print(f"Train DataLoader size: {len(train_loader)}")
    print(f"Test DataLoader size: {len(test_loader)}")
    for batch in train_loader:
        print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["labels"].shape)
        break