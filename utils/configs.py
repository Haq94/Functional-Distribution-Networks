
baseline_params ={'model_dict': 
{
    'IC_FDNet' :       {'hidden_dim': 23, 'hyper_hidden_dim': 6},      # ≈1004 params
    'LP_FDNet':        {'hidden_dim': 24, 'hyper_hidden_dim': 5},      # ≈1011
    'HyperNet':        {'hidden_dim': 25, 'hyper_hidden_dim': 9},      # ≈1012
    'BayesNet':        {'hidden_dim': 166},                            # ≈998
    'GaussHyperNet':   {'hidden_dim': 24, 'hyper_hidden_dim': 5, 'latent_dim': 9},  # ≈994
    'MLPNet':          {'hidden_dim': 333, 'dropout_rate': 0.1},       # ≈1000
    'MLPDropoutNet':   {'hidden_dim': 333, 'dropout_rate': 0.1},       # ≈1000
    'DeepEnsembleNet': {'hidden_dim': 33, 'dropout_rate': 0.1, 'num_models': 10,
                        'ensemble_seed_list': [0,1,2,3,4,5,6,7,8,9]}    # ≈1000 total
},
'epochs': 400,
'ensemble_epochs': 40,
'MC_train': 1,
'MC_val': 50,
'MC_test': 100,
'checkpoint_dict': {
    'stoch': {
    'metric_str': 'crps',
    'region_interp': (-1,1),
    'min_or_max': 'min',
    'interp_or_extrap': 'interp'
    },
    'det': {
    'metric_str': 'mse',
    'region_interp': (-1,1),
    'min_or_max': 'min',
    'interp_or_extrap': 'interp'
    }
    },
'linear_beta_scheduler': {"beta_scheduler": 'linear', "warmup_epochs": 200, "beta_max": 1},
'cosine_beta_scheduler': {"beta_scheduler": 'cosine', "warmup_epochs": 200, "beta_max": 1},
'signmoid_beta_scheduler': {"beta_scheduler": 'sigmoid', "warmup_epochs": 200, "beta_max": 1},
'unity_beta_scheduler': {"beta_scheduler": 'constant', "warmup_epochs": 200, "beta_max": 1},
'zero_beta_scheduler': {"beta_scheduler": 'constant', "warmup_epochs": 200, "beta_max": 1},
'region': (-10,10),
'region_interp': (-1,1),
'n_train': 1024,
'n_test': 2001,
'n_val_interp': 256,
'n_val_extrap': 256,
'plot_dict': {
    "Single": [],
    "Overlay": []
    }
}
