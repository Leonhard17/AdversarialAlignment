# File used to save model configuration parameters for hyperparameter sweeps

# NOTE: Just some random parameters for demonstration purposes.
# -----------------------------------------------------------------------------
# Weights & Biases hyperparameter sweep configuration
# -----------------------------------------------------------------------------
# Use Bayesian search over a subset of the most sensitive dims.
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "epoch_avg_loss",
        "goal": "minimize"
    },
    "parameters": {
        # optimizer params
        "lr": {
            "distribution": "log_uniform",
            "min": 1e-5,
            "max": 1e-3
        },
        "weight_decay": {
            "values": [0.0, 1e-6, 1e-4]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },

        # GNN over each head‑graph
        "gnn_hidden_dim": {
            "values": [64, 128, 256]
        },
        "gnn_embedding_dim": {
            "values": [32, 64, 128]
        },

        # Compressor MLP (per layer)
        "compression_hidden_dim": {
            "values": [256, 512, 1024]
        },
        "compression_dim": {
            "values": [64, 128, 256]
        },

        # Layer‑aggregation Transformer
        "agg_hidden_dim": {
            "values": [256, 512, 1024]
        },
        "agg_heads": {
            "values": [2, 4, 8]
        },
        "agg_layers": {
            "values": [2, 4, 6]
        },

        # Reward‑predictor Transformer
        "reward_hidden_dim": {
            "values": [256, 512, 1024]
        },
        "reward_heads": {
            "values": [2, 4, 8]
        },
        "reward_layers": {
            "values": [2, 4, 6]
        },
        "reward_ff_dim": {
            "values": [512, 1024, 2048]
        },

        # regularization
        "dropout": {
            "values": [0.0, 0.1, 0.2, 0.3]
        },

        # optional gradient clipping norm
        "grad_clip_norm": {
            "values": [0.0, 1.0, 5.0]
        }
    }
}