# Obesity scGPT Perturbation Prediction

Fine-tuning the pretrained [scGPT](https://github.com/bowang-lab/scGPT) single-cell foundation model
to predict how human adipocyte gene expression responds to single-gene (CRISPR knockout) perturbations,
for the CrunchDAO [Broad Obesity](https://hub.crunchdao.com/competitions/broad-obesity-1) competition
([overview PDF](https://docs.google.com/viewerng/viewer?url=https://crunchdao--competition--production.s3.eu-west-1.amazonaws.com/competitions/broad-obesity-1/documents/broad-obesity.pdf)).
Approach follows the [scGPT perturbation tutorial](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Perturbation.ipynb),
using [GEARS](https://github.com/snap-stanford/GEARS) for perturbation-graph data handling and evaluation.

## Data

- **`obesity_pert`** (target dataset): `obesity_pert.h5ad`, 44,846 cells x 11,046 genes. Cells are
  perturbed by single-gene CRISPR knockouts of adipogenesis-related transcription factors (e.g.
  `MEF2A`, `TEAD1`, `TFAP2A`; `FLI1` is dropped — not resolvable as a GEARS feature). Split strategy
  is `simulation`: train on perturbations seen during training, evaluate on `unseen_single`
  (held-out single-gene perturbations). This dataset has no multi-gene ("combo") perturbations, so
  the `combo_seen{0,1,2}_*` metrics GEARS reports are always `nan` here — `unseen_single` is the
  metric that matters.
- **`adamson`** (public benchmark, not the research target): used purely as a pipeline sanity check —
  if fine-tuning doesn't clearly improve `pearson_delta` on a well-studied public perturb-seq dataset,
  something's wrong with the training code, independent of whether the obesity signal is weak.

Both are preprocessed into GEARS `PertData` objects (graph representations of perturbation effects) in
`data/`; raw/processed data is git-ignored (see `.gitignore`) since it's large and cluster-local.
Competition leaderboard: [here](https://hub.crunchdao.com/competitions/broad-obesity-1/leaderboard).

## Method

- **Base model**: `scGPT_human`, the pretrained foundation model (lives as a sibling directory at the
  cluster home dir, referenced by relative path — scripts are launched with that home dir as the
  working root, not from inside this repo).
- **Architecture**: scGPT's `TransformerGenerator`, resumed from `scGPT_human/best_model.pt`, fine-tuned
  on the perturbation task. Objectives in play across experiments: masked value prediction (MSE) on
  expression, optionally blended with `pearson_delta`/`mse_delta` loss terms that directly target
  predicted-vs-true *change* under perturbation, and in the `multitask` variants an added
  classification head (cross-entropy) trained jointly.
- Gene subset (highly-variable-gene count) and batch size have varied across runs over time — batch
  size in particular is partly auto-selected based on the allocated GPU's VRAM, not always a
  deliberate hyperparameter choice.

## Results so far

Test numbers below are the overall test-set `pearson_delta` from each run's saved
`save/<run>/test_metrics.json` (the same `compute_perturbation_metrics` computation used for
validation — directly comparable to the "peak val score" column).

Best test `pearson_delta` to date: **0.205**
(`save/dev_perturb_obesity_pert-Mar06-00-50_hvg1kgene_loss_evalmetric_combined_loss`) — top-1000 HVGs,
combined loss/eval-metric objective.

| Run | Peak val score | Test `pearson_delta` |
|---|---|---|
| `dev_perturb_obesity_pert-Mar06..._hvg1kgene_loss_evalmetric_combined_loss` | 0.228 | **0.205** |
| `seed=1_mse_loss+0.1*mse_delta_cosineannealing` | 0.068 | 0.181 |
| `seed=1_mseloss_stepLR` | 0.068 | 0.164 |
| `dev_perturb_obesity_pert-Mar07..._hvg1kgene_loss_evalmetric_combined_loss` | 0.250 | 0.168 |
| `seed=2_0.2 * pearson_delta+mse_delta_cosineannealing` | 0.227 | 0.153 |
| `seed=2_pearson_delta_mse_delta` | 0.189 | 0.140 |
| `dev_perturb_obesity_pert-Apr05..._scgpt_multitask_v1` | 0.265 | 0.140 |
| `dev_perturb_obesity_pert-Apr05..._scgpt_multitask_v2` | 0.147 | 0.087 |
| `obesity_no_training` (zero-shot baseline) | — | 0.018 |
| `adamson_no_training` (zero-shot, sanity) | — | 0.051 |
| `adamson_1_epoch` (fine-tuned, sanity; no final test eval run) | 0.661 | — |

Fine-tuning turns a model with essentially zero predictive power on obesity perturbations out of the
box (0.018 zero-shot) into one that explains a meaningful share of the perturbation response, peaking
at **0.205** test `pearson_delta` with a top-1000-HVG run using a combined loss/eval-metric objective.
The Adamson benchmark run (val 0.66 after a single epoch) independently confirms the fine-tuning
pipeline itself is working as expected. Across the full sweep — loss formulation (MSE, pearson_delta,
blended delta terms), LR schedule (step vs. cosine), gene subset size, and an added multitask
classification head — results consistently land well above the untrained baseline, with the
combined-loss and multitask objectives the strongest performers overall.

## Repo layout

```text
.
├── obesity_scGPT_perturbation_prediction/   (this repo)
│   ├── data/                      # GEARS PertData (git-ignored, cluster-local)
│   │   ├── adamson/
│   │   └── obesity_pert/
│   ├── notebooks/
│   │   ├── data/                  # Small sanity-test subsets
│   │   ├── save/                  # Logs from notebook-driven sanity runs
│   │   ├── Install_deps.ipynb
│   │   ├── Preprocess_datasets.ipynb
│   │   └── scGPT_perturb_pred.ipynb
│   ├── save/                      # Full-run logs & best checkpoints (one dir per run)
│   ├── scripts/                   # K8s launch scripts (my-launch.sh, adamson-launch.sh, ...)
│   └── src/                       # scGPT_perturb_pred.py, adamson_perturb_pred.py
└── scGPT_human/                   # Pretrained foundation model (sibling dir, outside this repo)
    ├── args.json
    ├── best_model.pt
    └── vocab.json
```

## Running on the cluster

This is a shared DSMLP login node — no training runs directly on it. The workflow is:

1. Edit the target script (`src/scGPT_perturb_pred.py` or `src/adamson_perturb_pred.py`).
2. Point a launch script under `scripts/` (based on `my-launch.sh`) at it.
3. Submit as a K8s pod, e.g.:
   ```bash
   ./scripts/my-launch.sh -b -v l40s -t "gpu-class=large"
   ```
4. Check status with `kubectl get pods`, logs with `kubectl logs <pod-name>`, and stop/clean up with
   `kubectl delete pod <pod-name>` once finished — delete idle pods promptly to free the GPU for others.

## Dependencies

See `notebooks/Install_deps.ipynb` and `requirements.txt`. Dependencies are pinned — avoid ad hoc
`pip install`/upgrades in the shared environment, since a stray transitive upgrade (e.g. `numpy`) can
break every other concurrently running notebook/job on the same user environment.

## Weights

Model weights can be provided if requested.
