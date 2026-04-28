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
│   │   ├── data/
│   │   │   └── obesity_pert_sanity/
│   │   │       ├── data_pyg/
│   │   │       └── splits/
│   │   ├── save/                  # Local notebook experiment logs
│   │   │   └── dev_perturb_obesity_pert_sanity-*/
│   │   ├── Preprocess.ipynb
│   │   └── scGPT_analysis.ipynb
│   ├── save/                      # Main training outputs (Best Weights Only)
│   │   ├── multitask_v1/
│   │   ├── multitask_v2/
│   │   ├── loss_variant_seeds/
│   │   └── [Experimental_Trials...]
│   ├── scripts/                   # Cluster launch & sbatch scripts
│   └── src/                       # Core Python modules & model logic
│
└── scGPT_human/                   # Foundation Model (Pre-trained)
    ├── args.json                  # Config
    ├── model.pt                   # Weights (~500MB+)
    └── vocab.json                 # Gene Vocabulary
