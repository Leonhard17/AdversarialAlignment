import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb

from AttentionDataset.AttentionDataset import create_attention_loader
from Model.full_model               import FullAdversarialAlignmentModel
from config                         import secondary_model_config
from sweep_config                   import sweep_config

# how many epochs per run
num_epochs = 7

def train():
    # this context manager sets up wandb.config for us
    with wandb.init():
        config = wandb.config

        # parameters for gradeint 
        n_iter   = secondary_model_config["num_iterations"]
        n_layer  = secondary_model_config["num_layers"]
        n_head   = secondary_model_config["num_heads"]

        # 1) load data every run
        base = Path(__file__).resolve().parent.parent
        pkl_path = base / "AttentionDataset" / "data" / "attention_dataset.pkl"
        attentions, rewards = pickle.load(open(pkl_path, "rb"))
        loader = create_attention_loader(
            attentions, rewards, batch_size=config.batch_size, shuffle=True
        )

        # 2) merge sweep params into your model config
        model_cfg = secondary_model_config.copy()
        # merge sweep config into model config
        for key in list(model_cfg.keys()):
            if key in config:      
                model_cfg[key] = config[key]

        # 3) instantiate & move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullAdversarialAlignmentModel(**model_cfg).to(device)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=config.lr)

        # 4) training loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for attention_batches, rewards in tqdm(
                loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=80
            ):
                # move rewards
                rewards = rewards.to(device).float()

                # move each graph batch to device
                for iteration_layers in attention_batches:
                    for layer_heads in iteration_layers:
                        for i, g in enumerate(layer_heads):
                            layer_heads[i] = g.to(device)

                # forward & loss
                preds = model(attention_batches).squeeze()
                loss = criterion(preds, rewards)
                epoch_loss += loss.item()

                # backprop
                optimizer.zero_grad()
                loss.backward()

                # Gradient normalization
                # Because using some models multiple times in a batch
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

                # batch‐level logging
                wandb.log({
                    "batch_loss":      loss.item(),
                    "preds_mean":      preds.mean().item(),
                    "preds_std":       preds.std().item(),
                    "targets_mean":    rewards.mean().item(),
                    "targets_std":     rewards.std().item(),
                    "grad_norm":       sum(p.grad.data.norm(2).item()**2
                                           for p in model.parameters() 
                                           if p.grad is not None)**0.5,
                })

            # epoch‐level checkpointing
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} — Avg Loss: {avg_loss:.6f}")
            wandb.log({"epoch_avg_loss": avg_loss})
            run_name = wandb.run.name 
            ckpt_dir = base / "alignment_models" / f"{run_name}_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_dir)
        
        
if __name__ == "__main__":
    # launch the sweep
    sweep_id = wandb.sweep(sweep_config, project="adversarial-alignment")
    # run up to 20 trials (you can change the count or remove it for infinite)
    wandb.agent(sweep_id, function=train, count=20)
