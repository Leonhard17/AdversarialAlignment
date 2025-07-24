from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_scheduler
from MathDataset import create_data_loaders

def train_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        inputs = batch["input_ids"].to(device)
        masks  = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            inputs = batch["input_ids"].to(device)
            masks  = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            total_loss += outputs.loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def finetune_model(
    problem_file: Path,
    solution_file: Path,
    save_dir: Path,
    base_model: str = "gpt2",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
):
    # 1) Load data & tokenizer (with pad_token added)
    train_loader, val_loader, tokenizer = create_data_loaders(
        problem_filename=problem_file,
        solution_filename=solution_file,
        batch_size=batch_size,
        tokenizer_name=base_model,
    )

    # 2) Make sure pad_token is present & resize model embeddings
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # load model & resize token embeddings
    model = GPT2LMHeadModel.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    # tell the model which token ID is the pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3) Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # 4) Move to GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss   = eval_epoch(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 5) Save both tokenizer & model *after* each epoch
        ckpt_dir = save_dir / f"finetuned_multiplication_{base_model}_epoch_{epoch}"
        ckpt_dir.mkdir(exist_ok=True)
        tokenizer.save_pretrained(ckpt_dir)
        model.save_pretrained(ckpt_dir)

    return history


if __name__ == "__main__":
    BASE_DIR      = Path(__file__).resolve().parent
    problem_file  = BASE_DIR / "MathDataset" / "data" / "math_problems.txt"
    solution_file = BASE_DIR / "MathDataset" / "data" / "math_solutions.txt"
    save_dir      = BASE_DIR / "finetuned_models"

    history = finetune_model(
        problem_file=problem_file,
        solution_file=solution_file,
        save_dir=save_dir,
        base_model="gpt2",
        batch_size=32,
        num_epochs=7,
        learning_rate=5e-5,
    )
    print("\nFine-tuning complete.")


# Optional: Uncomment to use argparse for command line arguments
# from argparse import ArgumentParser

# parser = ArgumentParser(description="Fine-tune GPT-2 on math dataset")
# parser.add_argument("--problems", type=Path, required=True, help="Path to math_problems.txt")
# parser.add_argument("--solutions", type=Path, required=True, help="Path to math_solutions.txt")
# parser.add_argument("--out_dir", type=Path, required=True, help="Directory to save checkpoints")
# parser.add_argument("--model", type=str, default="gpt2", help="Pretrained model name")
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--epochs", type=int, default=3)
# parser.add_argument("--lr", type=float, default=5e-5)
# args = parser.parse_args()

# history = finetune_model(
#     problem_file=args.problems,
#     solution_file=args.solutions,
#     save_dir=args.out_dir,
#     model_name=args.model,
#     batch_size=args.batch_size,
#     num_epochs=args.epochs,
#     learning_rate=args.lr,
# )
# print("Fine-tuning complete.")
