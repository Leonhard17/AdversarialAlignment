# File used to save model configuration parameters for hyperparameter sweeps

# NOTE: Just some random parameters for demonstration purposes.
sweep_config = {
    'method': 'bayes',  # could also use 'grid' or 'random'
    'metric': {
        'name': 'epoch_avg_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'dropout': {
            'values': [0.1, 0.2, 0.3]
        },
        'compression_dim': {
            'values': [4, 8, 16]
        },
        'agg_hidden_dim': {
            'values': [4, 8, 16]
        },
        'lr': {
            'values': [1e-5, 1e-4, 1e-3]
        }
    }
}
