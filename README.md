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

Note: if you do not have the pre-built `dataset_cache/` files checked into your
clone (these are large and often omitted from git), recreate the cache by
running the dataset creation notebook `notebooks/01_dataset_creation.ipynb`,
which contains the full data collection and preprocessing steps used to build
the training/validation/test splits and normalization statistics.

## Calibration diagnostics

This fork adds a small calibration tooling suite used to validate probabilistic
coverage and reliability of CFM ensemble forecasts. Key script:

- `diagnose_model.py`: lightweight per-checkpoint diagnostics useful for quick
   local checks (defaults: `--n_subset=128`, `--n_samples=32`,
   `--radii 50 100 200 400 800`). It produces per-radius reliability plots,
   normalized cumulative calibration plots, and CSV summaries.

Outputs (examples):

- `calibration_summary.csv`, `calibration_regime_summary.csv`
- `equal_width_bins_r{r}.csv`, `equal_count_bins_r{r}.csv`
- `raw/raw_r{r}.csv` and `raw/raw_r{r}.npz` (contains `p_hat`, `y_true`, bin
   ids and `flight_id` when available)
- `plots/reliability_r{r}.png`, `plots/cumulative_r{r}.png`
- `pit_summary.csv`, `coverage_summary.csv`, `score_summary.csv`

Quick smoke-test command you can run locally (small, CPU/MPS-friendly):

```bash
# tiny smoke test (4 samples, 2 ensemble members)
.venv/bin/python diagnose_model.py \
   --ckpt models/cfm_base.pt --n_subset 4 --n_samples 2 \
   --out-dir eval_smoke --batch-size 2 --n_steps 8 --nbins 5
```

For a full GPU-backed evaluation use a larger subset and sample count, for
example via SLURM or the cluster scheduler.

## Notes about training

- The base CFM backbone was trained on the cached dataset and produces
   probabilistic trajectories by sampling the learned conditional flow. Default
   script configs point to `models/cfm_base.pt`.

---