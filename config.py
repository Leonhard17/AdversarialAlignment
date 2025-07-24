# File used to save model configuration parameters
# The parameters found here only are used for demonstration purposes
# both models are to small to produce any meaningful results

# ----------------------
# Primary model config
# ----------------------
# This is the transformer (LLM) which produces the data to be supervised.
# These settings are used for pre-training and then for generating attention graphs.

primary_model_config = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768
}

# -----------------------------------------------------------------------------
# Secondary (reward‑predictor) model configuration
# -----------------------------------------------------------------------------
# This network reads the sequence of attention‑graph embeddings produced
# by the primary GPT‑2 model over multiple decoding iterations, and predicts
# a scalar “reward” (e.g. solution accuracy or uncertainty).
secondary_model_config = {
    # How many decoding iterations (time‑steps) of attention graphs to process:
    "num_iterations": 5,  

    # Mirror the primary LLM’s architecture:
    "num_layers": primary_model_config["n_layer"],   # e.g. 12 transformer layers
    "num_heads":  primary_model_config["n_head"],    # e.g. 12 attention heads

    # ─── GNN per head ─────────────────────────────────────────────────────────
    "gnn_hidden_dim":    128,   # doubled from 64  
    "gnn_embedding_dim": 64,    # doubled from 32  

    # ─── Per-layer compressor MLP ────────────────────────────────────────────
    "compression_hidden_dim": 512,  # doubled from 256  
    "compression_dim":        128,  # doubled from 64   

    # ─── Layer-aggregation transformer ────────────────────────────────────────
    "agg_hidden_dim": 512,   # doubled from 256  
    "agg_heads":      8,     # doubled from 4    
    "agg_layers":     8,     # doubled from 4    

    # ─── Reward-predictor transformer ────────────────────────────────────────
    "reward_hidden_dim": 1024,   # doubled from 512  
    "reward_heads":      16,     # doubled from 8    
    "reward_layers":     12,     # doubled from 6    
    "reward_ff_dim":     4096,   # doubled from 2048

    # Dropout to regularize all Transformer & MLP layers
    "dropout": 0.1
}

