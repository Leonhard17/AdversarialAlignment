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
        "--batch-size", type=int, default=128,
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
    
    n_iter   = secondary_model_config["num_iterations"]
    n_layer  = primary_model_config["n_layer"]
    n_head   = primary_model_config["n_head"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load data
    with open(args.data_pkl, "rb") as f:
        attentions, rewards = pickle.load(f)

    loader = create_attention_loader(
        attentions, rewards,
        batch_size=args.batch_size,
        shuffle=True,
        device=device
    )

    # 2) Init model with your secondary config
    model_cfg = secondary_model_config.copy()
    model = FullAdversarialAlignmentModel(**model_cfg)
    model.to(device)

    # 3) Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Folder for saving models
    Base_DIR = Path(__file__).resolve().parent.parent
    (Base_DIR / "alignment_models").mkdir(exist_ok=True)

    # 5) Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        epoch_loss = 0.0

        for attention_batches, rewards in tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80
        ):
            # forward + loss
            preds = model(attention_batches).squeeze()
            loss = criterion(preds, rewards)
            epoch_loss += loss.item()

            # backward + step
            # TODO: Grdient normalization
            # Because using some models multiple times in a batch
            optimizer.zero_grad()
            loss.backward()
            
            gnn_calls  = n_iter * n_layer * n_head
            comp_calls = n_iter * n_layer
            agg_calls = n_iter

            # scale down GNN gradients
            for p in model.gnn.parameters():
                if p.grad is not None:
                    p.grad.data.div_(gnn_calls)

            # scale down each compressor’s gradients
            for compressor in model.compressors:
                for p in compressor.parameters():
                    if p.grad is not None:
                        p.grad.data.div_(comp_calls)

            #scale down aggregation encoder gradients
            for p in model.aggregation_encoder.parameters():
                if p.grad is not None:
                    p.grad.data.div_(agg_calls)

            
            optimizer.step()

            

        # epoch metrics
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.6f}")

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