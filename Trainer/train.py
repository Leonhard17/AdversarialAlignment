import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb

from AttentionDataset.AttentionDataset import create_attention_loader
from Model.full_model               import FullAdversarialAlignmentModel

# import your configs
from config import primary_model_config, secondary_model_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Train FullAdversarialAlignmentModel on attention/reward data"
    )
    p.add_argument(
        "--data-pkl", type=Path, required=True,
        help="Path to attention_dataset.pkl"
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    p.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    p.add_argument(
        "--project", type=str, default="adversarial-alignment",
        help="W&B project name"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load data
    with open(args.data_pkl, "rb") as f:
        attentions, rewards = pickle.load(f)

    loader = create_attention_loader(
        attentions, rewards,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 2) Init model with your secondary config
    model_cfg = secondary_model_config.copy()
    model = FullAdversarialAlignmentModel(**model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 4) Start W&B and log config
    wandb.init(
        project=args.project,
        config={
            "primary_model": primary_model_config,
            "secondary_model": secondary_model_config,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
        }
    )
    # Folder for saving models
    Base_DIR = Path(__file__).resolve().parent.parent
    (Base_DIR / "alignment_models").mkdir(exist_ok=True)

    # 5) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for attention_batches, rewards in tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80
        ):
            # move rewards & graphs to device
            rewards = rewards.to(device).float()
            for it in attention_batches:
                for layer_heads in it:
                    for i, g in enumerate(layer_heads):
                        layer_heads[i] = g.to(device)

            # forward + loss
            preds = model(attention_batches).squeeze()
            loss = criterion(preds, rewards)
            epoch_loss += loss.item()

            # backward + step
            # TODO: Grdient normalization
            # Because using some models multiple times in a batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # per‑batch logging
            wandb.log({
                "batch_loss": loss.item(),
                "preds_mean": preds.mean().item(),
                "preds_std": preds.std().item(),
                "targets_mean": rewards.mean().item(),
                "targets_std": rewards.std().item(),
            })

            # gradient norm
            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            wandb.log({"grad_norm": total_norm})

        # epoch metrics
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.6f}")
        wandb.log({"epoch_avg_loss": avg_loss})

        # save checkpoint
        ckpt_path = Path(f"alignment_model_epoch_{epoch}.pt")
        ckpt_path = Base_DIR / "alignment_models" / ckpt_path
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # 6) final save
    final_path = Path("alignment_model_final.pt")
    final_path = Base_DIR / "alignment_models" / final_path
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()