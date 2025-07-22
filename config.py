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

# ----------------------
# Secondary model config
# ----------------------
# This is the architecture which processes the attention graphs
# from the primary model and learns to predict rewards or uncertainty.

secondary_model_config = {
    "num_iterations": 5, # number of iterations of primary model output
    "num_layers": primary_model_config["n_layer"],
    "num_heads": primary_model_config["n_head"],
    "gnn_hidden_dim": 16,
    "gnn_embedding_dim": 8,
    "compression_hidden_dim": 32,
    "compression_dim": 4,
    "agg_hidden_dim": 4,
    "agg_heads": 2,
    "agg_layers": 2,
    "reward_hidden_dim": 24,
    "reward_heads": 2,
    "reward_layers": 2,
    "reward_ff_dim": 64,
    "dropout": 0.1
}
