## Project Structure

```text
.
├── obesity_scGPT_perturbation_prediction/
│   ├── data/
│   │   ├── adamson/
│   │   │   ├── data_pyg/          # Processed PyTorch Geometric graphs
│   │   │   └── splits/            # Train/Val/Test split files
│   │   └── obesity_pert/
│   │       ├── data_pyg/
│   │       └── splits/
│   ├── notebooks/
│   │   ├── data/                  # Sanity check datasets
│   │   ├── save/                  # Notebook-specific logs
│   │   ├── Preprocess.ipynb
│   │   └── scGPT_analysis.ipynb
│   ├── save/                      # Main training outputs & best weights
│   ├── scripts/                   # Cluster launch scripts (.sh)
│   └── src/                       # Model logic and utility modules
└── scGPT_human/                   # Foundation Model (Pre-trained)
    ├── args.json
    ├── model.pt
    └── vocab.json
```

## Dependencies
Check out notebooks/install_deps.ipynb for details


## Data
Data is from [this](https://hub.crunchdao.com/competitions/broad-obesity-1/leaderboard) competition. 

## Weights
Model weights can be provided if requested. 
