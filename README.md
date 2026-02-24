# Generative Flight Trajectory Prediction

Probabilistic aircraft trajectory prediction using Conditional Flow Matching (CFM). Given one minute of observed flight data, generates multiple possible 1-minute future trajectories for uncertainty-aware safety analysis in Air Traffic Management.

## Notebooks

- `OSN_paper_training_1min.ipynb` - Trains the CFM model on ADS-B data
- `osn_paper_results_1min.ipynb` - Evaluates model performance with metrics and visualizations
- `osn_paper_crm.ipynb` - Showcases collision risk modeling for flight conflicts


## Data & model access

To reproduce the results notebooks, download the artifacts (dataset, cache, and model) from the Zenodo record:
- DOI: https://doi.org/10.5281/zenodo.17869284
- Direct record: https://zenodo.org/records/17869284

Steps:
1) Download the archive from “Download all” on the Zenodo page (or fetch individual files).
2) Extract into the repository root so paths match expectations:
   - `trajs_LSAS_filtered.parquet` at the repo root
   - `models/model_1min.pt`
   - `dataset_cache/` with the `ecec4b007a021fa3.*` files (and associated `.npy/.parquet/.json`)
3) Run the notebooks.

## Intent-conditioned calibration experiment

This fork adds a focused experiment to test whether explicit kinematic intent
conditioning can improve probabilistic calibration (PIT), especially in lateral
axes.

### What changes

- Keep the original CFM backbone unchanged.
- Extend the context with intent descriptors derived from the same 60 s history:
  - 12 continuous kinematic intent features
  - 25 one-hot intent classes (vertical x lateral phase)
- Evaluate with paper-style plots and calibration diagnostics:
  - MAE/RMSE vs horizon (model mean, best-of-S, CV baseline)
  - PIT histograms and uniformity scores
  - Over-dispersion index across horizon