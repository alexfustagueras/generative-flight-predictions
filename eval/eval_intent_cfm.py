#!/usr/bin/env python3
"""
Evaluate Intent-Conditioned CFM model: PIT histograms, calibration, "spaghetti plots"

Produces evaluation on the test set (or a subset) and saves figures.

Usage:
    python eval_intent_cfm.py [--n-eval 2000] [--n-samples 64] [--batch-size 64]
"""

import sys, json, argparse, math, time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

GFP_DIR = SCRIPT_DIR.parent / "generative-flight-predictions"
sys.path.insert(0, str(GFP_DIR))

from model_intent import IntentFlowMatchingModel, get_model_config
from intent import (
    compute_intent_features_batch, classify_intent_batch,
    intent_index_to_onehot, denormalize_history, intent_name,
)
from utils.inference_utils import sample_future_heun, denorm_seq_to_global

def load_intent_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )
    cfg = ckpt.get("model_cfg", get_model_config())
    model = IntentFlowMatchingModel(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

def build_extended_context(X_norm, C_norm, feat_mean, feat_std,
                           intent_feat_mean, intent_feat_std):
    """Build 45-dim context: [C_8, intent_12_norm, onehot_25]."""
    X_phys = denormalize_history(X_norm, feat_mean, feat_std)
    intent_feats = compute_intent_features_batch(X_phys)
    intent_labels = classify_intent_batch(X_phys)

    intent_feats_norm = ((intent_feats - intent_feat_mean) / intent_feat_std).astype(np.float32)
    onehot = intent_index_to_onehot(intent_labels, 25)

    C_ext = np.concatenate([C_norm, intent_feats_norm, onehot], axis=1)
    return C_ext.astype(np.float32), intent_labels

@torch.no_grad()
def pit_values(y_samples, y_true):
    """PIT values per axis. y_samples: (S,B,T,3), y_true: (B,T,3)."""
    S, B, T, D = y_samples.shape
    D = min(3, D)
    Y_sorted = torch.sort(y_samples[..., :D], dim=0).values
    pits = torch.empty(B, T, D, device=y_samples.device, dtype=torch.float32)
    for d in range(D):
        cmp = Y_sorted[..., d] <= y_true[..., d]
        ranks = cmp.sum(dim=0)
        pits[..., d] = (
            ranks.to(torch.float32) + torch.rand_like(ranks, dtype=torch.float32)
        ) / (S + 1.0)
    return pits

def pit_ks_distance(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance to Uniform[0,1]"""
    x = np.sort(np.asarray(u, dtype=np.float64).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - x)
    d_minus = np.max(x - (i - 1) / n)
    return float(max(d_plus, d_minus))

@torch.no_grad()
def evaluate(model, X_test, Y_test, C_test_ext, C_test_8,
             feat_mean, feat_std, ctx_mean, ctx_std,
             device, n_eval=2000, n_samples=64, batch_size=64, n_steps=64, dt_seconds=5.0):
    """Run evaluation: generate ensembles and compute PIT + errors."""
    N = min(n_eval, X_test.shape[0])
    T = Y_test.shape[1]

    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], size=N, replace=False)

    all_pits = []
    all_ade_mean = []
    all_ade_single = []
    all_fde_mean = []
    all_spreads = []
    # Per-horizon deterministic errors (ensemble mean predictor)
    all_mae3d_mean = []
    all_mae_xy_mean = []
    all_mae_z_mean = []
    all_mse3d_mean = []
    all_mse_xy_mean = []
    all_mse_z_mean = []
    # Best-of-S error curves
    all_mae3d_best = []
    all_mae_xy_best = []
    all_mae_z_best = []
    all_mse3d_best = []
    all_mse_xy_best = []
    all_mse_z_best = []
    # Constant velocity baseline curves
    all_mae3d_cv = []
    all_mae_xy_cv = []
    all_mae_z_cv = []
    all_mse3d_cv = []
    all_mse_xy_cv = []
    all_mse_z_cv = []
    n_done = 0

    t0 = time.time()
    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        batch_idx = idx[i0:i1]
        B = len(batch_idx)

        x_hist = torch.from_numpy(np.array(X_test[batch_idx])).float().to(device)
        ctx_ext = torch.from_numpy(np.array(C_test_ext[batch_idx])).float().to(device)
        ctx_8 = torch.from_numpy(np.array(C_test_8[batch_idx])).float().to(device)
        y_true_norm = torch.from_numpy(np.array(Y_test[batch_idx])).float().to(device)

        y_samp_norm = sample_future_heun(
            model,
            x_hist.repeat(n_samples, 1, 1),
            ctx_ext.repeat(n_samples, 1),
            T_out=T, n_steps=n_steps, G=1.0,
        ).view(n_samples, B, T, -1)

        y_samp_glob = denorm_seq_to_global(
            y_samp_norm.reshape(-1, T, 7),
            ctx_8.repeat(n_samples, 1),
            feat_mean, feat_std, ctx_mean, ctx_std,
        ).view(n_samples, B, T, -1)

        y_true_glob = denorm_seq_to_global(
            y_true_norm, ctx_8, feat_mean, feat_std, ctx_mean, ctx_std,
        )
        x_hist_glob = denorm_seq_to_global(
            x_hist, ctx_8, feat_mean, feat_std, ctx_mean, ctx_std,
        )

        pits_bt3 = pit_values(y_samp_glob[..., :3], y_true_glob[..., :3])
        all_pits.append(pits_bt3.cpu())

        y_mean = y_samp_glob.mean(dim=0)
        ade_mean = torch.linalg.norm(y_mean[..., :3] - y_true_glob[..., :3], dim=-1).mean(dim=1)
        fde_mean = torch.linalg.norm(y_mean[..., :3] - y_true_glob[..., :3], dim=-1)[:, -1]
        ade_single = torch.linalg.norm(y_samp_glob[0, ..., :3] - y_true_glob[..., :3], dim=-1).mean(dim=1)

        # Per-horizon absolute/squared errors for MAE/RMSE curves
        err = y_mean[..., :3] - y_true_glob[..., :3]  # (B, T, 3)
        err_xy = err[..., :2]
        err_z = err[..., 2]

        e3d = torch.linalg.norm(err, dim=-1)  # (B, T)
        exy = torch.linalg.norm(err_xy, dim=-1)  # (B, T)
        ez = torch.abs(err_z)  # (B, T)

        all_mae3d_mean.append(e3d.cpu())
        all_mae_xy_mean.append(exy.cpu())
        all_mae_z_mean.append(ez.cpu())
        all_mse3d_mean.append((e3d ** 2).cpu())
        all_mse_xy_mean.append((exy ** 2).cpu())
        all_mse_z_mean.append((ez ** 2).cpu())

        # Best-of-S (oracle sample at each horizon)
        err_s = y_samp_glob[..., :3] - y_true_glob.unsqueeze(0)[..., :3]  # (S,B,T,3)
        e3d_s = torch.linalg.norm(err_s, dim=-1)  # (S,B,T)
        exy_s = torch.linalg.norm(err_s[..., :2], dim=-1)  # (S,B,T)
        ez_s = torch.abs(err_s[..., 2])  # (S,B,T)
        best3d = e3d_s.min(dim=0).values  # (B,T)
        bestxy = exy_s.min(dim=0).values
        bestz = ez_s.min(dim=0).values
        all_mae3d_best.append(best3d.cpu())
        all_mae_xy_best.append(bestxy.cpu())
        all_mae_z_best.append(bestz.cpu())
        all_mse3d_best.append((best3d ** 2).cpu())
        all_mse_xy_best.append((bestxy ** 2).cpu())
        all_mse_z_best.append((bestz ** 2).cpu())

        # Constant velocity baseline from last history state (global frame)
        p0 = x_hist_glob[:, -1, :3]   # (B,3)
        v0 = x_hist_glob[:, -1, 3:6]  # (B,3)
        t_steps = (
            torch.arange(1, T + 1, device=device, dtype=p0.dtype).view(1, T, 1)
            * float(dt_seconds)
        )
        y_cv = p0.unsqueeze(1) + v0.unsqueeze(1) * t_steps  # (B,T,3)
        err_cv = y_cv - y_true_glob[..., :3]
        e3d_cv = torch.linalg.norm(err_cv, dim=-1)
        exy_cv = torch.linalg.norm(err_cv[..., :2], dim=-1)
        ez_cv = torch.abs(err_cv[..., 2])
        all_mae3d_cv.append(e3d_cv.cpu())
        all_mae_xy_cv.append(exy_cv.cpu())
        all_mae_z_cv.append(ez_cv.cpu())
        all_mse3d_cv.append((e3d_cv ** 2).cpu())
        all_mse_xy_cv.append((exy_cv ** 2).cpu())
        all_mse_z_cv.append((ez_cv ** 2).cpu())

        spread = torch.sqrt((y_samp_glob[..., :3].var(dim=0)).sum(dim=-1))

        all_ade_mean.append(ade_mean.cpu())
        all_fde_mean.append(fde_mean.cpu())
        all_ade_single.append(ade_single.cpu())
        all_spreads.append(spread.cpu())

        n_done += B
        elapsed = time.time() - t0
        eta = elapsed / n_done * (N - n_done) if n_done > 0 else 0
        print(f"  [{n_done:>5d}/{N}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    pits_all = torch.cat(all_pits, dim=0)
    ade_mean_all = torch.cat(all_ade_mean)
    fde_mean_all = torch.cat(all_fde_mean)
    ade_single_all = torch.cat(all_ade_single)
    spread_all = torch.cat(all_spreads, dim=0)
    mae3d_mean_all = torch.cat(all_mae3d_mean, dim=0)
    mae_xy_mean_all = torch.cat(all_mae_xy_mean, dim=0)
    mae_z_mean_all = torch.cat(all_mae_z_mean, dim=0)
    rmse3d_mean_curve = torch.sqrt(torch.cat(all_mse3d_mean, dim=0).mean(dim=0))
    rmse_xy_mean_curve = torch.sqrt(torch.cat(all_mse_xy_mean, dim=0).mean(dim=0))
    rmse_z_mean_curve = torch.sqrt(torch.cat(all_mse_z_mean, dim=0).mean(dim=0))
    mae3d_best_curve = torch.cat(all_mae3d_best, dim=0).mean(dim=0)
    mae_xy_best_curve = torch.cat(all_mae_xy_best, dim=0).mean(dim=0)
    mae_z_best_curve = torch.cat(all_mae_z_best, dim=0).mean(dim=0)
    rmse3d_best_curve = torch.sqrt(torch.cat(all_mse3d_best, dim=0).mean(dim=0))
    rmse_xy_best_curve = torch.sqrt(torch.cat(all_mse_xy_best, dim=0).mean(dim=0))
    rmse_z_best_curve = torch.sqrt(torch.cat(all_mse_z_best, dim=0).mean(dim=0))
    mae3d_cv_curve = torch.cat(all_mae3d_cv, dim=0).mean(dim=0)
    mae_xy_cv_curve = torch.cat(all_mae_xy_cv, dim=0).mean(dim=0)
    mae_z_cv_curve = torch.cat(all_mae_z_cv, dim=0).mean(dim=0)
    rmse3d_cv_curve = torch.sqrt(torch.cat(all_mse3d_cv, dim=0).mean(dim=0))
    rmse_xy_cv_curve = torch.sqrt(torch.cat(all_mse_xy_cv, dim=0).mean(dim=0))
    rmse_z_cv_curve = torch.sqrt(torch.cat(all_mse_z_cv, dim=0).mean(dim=0))

    return {
        "pits": pits_all,
        "ade_mean": ade_mean_all,
        "fde_mean": fde_mean_all,
        "ade_single": ade_single_all,
        "spread": spread_all,
        "mae3d_mean_curve": mae3d_mean_all.mean(dim=0),
        "mae_xy_mean_curve": mae_xy_mean_all.mean(dim=0),
        "mae_z_mean_curve": mae_z_mean_all.mean(dim=0),
        "rmse3d_mean_curve": rmse3d_mean_curve,
        "rmse_xy_mean_curve": rmse_xy_mean_curve,
        "rmse_z_mean_curve": rmse_z_mean_curve,
        "mae3d_best_curve": mae3d_best_curve,
        "mae_xy_best_curve": mae_xy_best_curve,
        "mae_z_best_curve": mae_z_best_curve,
        "rmse3d_best_curve": rmse3d_best_curve,
        "rmse_xy_best_curve": rmse_xy_best_curve,
        "rmse_z_best_curve": rmse_z_best_curve,
        "mae3d_cv_curve": mae3d_cv_curve,
        "mae_xy_cv_curve": mae_xy_cv_curve,
        "mae_z_cv_curve": mae_z_cv_curve,
        "rmse3d_cv_curve": rmse3d_cv_curve,
        "rmse_xy_cv_curve": rmse_xy_cv_curve,
        "rmse_z_cv_curve": rmse_z_cv_curve,
        "n_eval": N,
        "n_samples": n_samples,
        "eval_indices": idx,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/cfm_intent_best.pt")
    parser.add_argument("--dataset-key", default="51397c8e8791f8ca")
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--dt-seconds", type=float, default=5.0,
                        help="Time between forecast steps, in seconds (default 5s).")
    parser.add_argument("--output", default="eval_results.npz")
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    ckpt_path = SCRIPT_DIR / args.ckpt
    intent_norm_path = ckpt_path.with_suffix(".intent_norm.json")

    print(f"Loading model from {ckpt_path}...")
    model, cfg = load_intent_model(ckpt_path, device)
    print(f"  context_dim={cfg['context_dim']}, params={sum(p.numel() for p in model.parameters()):,}")

    print(f"Loading intent norm stats from {intent_norm_path}...")
    intent_norm = json.loads(intent_norm_path.read_text())
    intent_feat_mean = np.array(intent_norm["intent_feat_mean"], dtype=np.float32)
    intent_feat_std = np.array(intent_norm["intent_feat_std"], dtype=np.float32)

    CACHE_DIR = SCRIPT_DIR / "dataset_cache"
    DSET_KEY = args.dataset_key
    FILES_DIR = CACHE_DIR

    key_info = json.loads((FILES_DIR / f"{DSET_KEY}.key.json").read_text())
    stats_key = key_info["stats_key"]
    norm_stats_path = FILES_DIR / f"{stats_key}.norm_stats.json"
    if not norm_stats_path.exists():
        legacy_files_dir = CACHE_DIR / f"{DSET_KEY}_FILES"
        if legacy_files_dir.exists():
            FILES_DIR = legacy_files_dir
            norm_stats_path = FILES_DIR / f"{stats_key}.norm_stats.json"
        else:
            norm_stats_path = CACHE_DIR / f"{stats_key}.norm_stats.json"
    norm_stats = json.loads(norm_stats_path.read_text())
    feat_mean = np.array(norm_stats["feat_mean"], dtype=np.float32)
    feat_std = np.array(norm_stats["feat_std"], dtype=np.float32)
    ctx_mean = np.array(norm_stats["ctx_mean"], dtype=np.float32)
    ctx_std = np.array(norm_stats["ctx_std"], dtype=np.float32)

    print(f"Loading test data from {FILES_DIR}...")
    X_test = np.load(FILES_DIR / f"{DSET_KEY}.X_test.npy", mmap_mode="r")
    Y_test = np.load(FILES_DIR / f"{DSET_KEY}.Y_test.npy", mmap_mode="r")
    C_test = np.load(FILES_DIR / f"{DSET_KEY}.C_test.npy", mmap_mode="r")
    print(f"  X_test: {X_test.shape}, Y_test: {Y_test.shape}, C_test: {C_test.shape}")

    print("Building extended context (computing intent features on test set)...")
    X_test_arr = np.array(X_test[:args.n_eval])
    C_test_arr = np.array(C_test[:args.n_eval])
    C_ext, intent_labels = build_extended_context(
        X_test_arr, C_test_arr, feat_mean, feat_std,
        intent_feat_mean, intent_feat_std,
    )
    print(f"  C_ext shape: {C_ext.shape}")

    print(f"\nRunning evaluation: n_eval={args.n_eval}, n_samples={args.n_samples}, "
          f"batch_size={args.batch_size}, n_steps={args.n_steps}")
    results = evaluate(
        model,
        X_test_arr, np.array(Y_test[:args.n_eval]),
        C_ext, C_test_arr,
        feat_mean, feat_std, ctx_mean, ctx_std,
        device=device,
        n_eval=args.n_eval,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        dt_seconds=args.dt_seconds,
    )

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  ADE (mean ensemble):   {results['ade_mean'].mean():.1f} m")
    print(f"  ADE (single sample):   {results['ade_single'].mean():.1f} m")
    print(f"  FDE (mean ensemble):   {results['fde_mean'].mean():.1f} m")
    print(f"  Spread (mean):         {results['spread'].mean():.1f} m")

    pits = results["pits"].numpy()
    print(f"\n  PIT uniformity check (ideal = 0.5):")
    pit_diag = {}
    ideal_std = 1.0 / math.sqrt(12.0)
    for d, name in enumerate(["x", "y", "z"]):
        p = pits[:, :, d].ravel()
        ks = pit_ks_distance(p)
        std = float(p.std())
        gap = abs(std - ideal_std)
        print(f"    {name}: mean={p.mean():.3f}, std={std:.3f}, KS={ks:.3f}, |std-ideal|={gap:.3f}")
        pit_diag[f"pit_{name}_mean"] = float(p.mean())
        pit_diag[f"pit_{name}_std"] = std
        pit_diag[f"pit_{name}_ks"] = ks
        pit_diag[f"pit_{name}_std_gap"] = gap

    # Over-dispersion index: spread / RMSE(ensemble-mean), by horizon.
    # >1 suggests over-dispersion, <1 suggests under-dispersion.
    spread_curve = results["spread"].mean(dim=0).numpy()
    rmse3d_curve = results["rmse3d_mean_curve"].numpy()
    odi_curve = spread_curve / np.maximum(rmse3d_curve, 1e-9)
    print("\n  Over-dispersion index (spread/RMSE3D, ideal ~1):")
    print(f"    +5s:  {odi_curve[0]:.3f}")
    print(f"    +15s: {odi_curve[2]:.3f}")
    print(f"    +30s: {odi_curve[5]:.3f}")
    print(f"    +60s: {odi_curve[-1]:.3f}")

    # Aggregate by lateral intent family to answer "straight vs turning?"
    # label mapping: intent_idx = 5*vertical + lateral, where lateral in [0..4]
    lat_phase = intent_labels % 5
    family_defs = {
        "Straight": np.isin(lat_phase, [0]),
        "Turning": np.isin(lat_phase, [1, 2]),
        "Roll-out": np.isin(lat_phase, [3, 4]),
    }
    print("\n  Intent family summary:")
    print("    family      n    ADE_mean   FDE_mean   spread   PITx_mean  PITy_mean")
    for fam, mask in family_defs.items():
        if mask.sum() == 0:
            continue
        ade_f = float(results["ade_mean"].numpy()[mask].mean())
        fde_f = float(results["fde_mean"].numpy()[mask].mean())
        spr_f = float(results["spread"].numpy()[mask].mean())
        pitx_f = float(pits[mask, :, 0].mean())
        pity_f = float(pits[mask, :, 1].mean())
        print(f"    {fam:<9} {int(mask.sum()):>4d}  {ade_f:>9.2f}  {fde_f:>9.2f}  {spr_f:>7.2f}   {pitx_f:>8.3f}  {pity_f:>8.3f}")

    out_path = SCRIPT_DIR / args.output
    np.savez_compressed(
        out_path,
        pits=pits,
        ade_mean=results["ade_mean"].numpy(),
        fde_mean=results["fde_mean"].numpy(),
        ade_single=results["ade_single"].numpy(),
        spread=results["spread"].numpy(),
        mae3d_mean_curve=results["mae3d_mean_curve"].numpy(),
        mae_xy_mean_curve=results["mae_xy_mean_curve"].numpy(),
        mae_z_mean_curve=results["mae_z_mean_curve"].numpy(),
        rmse3d_mean_curve=results["rmse3d_mean_curve"].numpy(),
        rmse_xy_mean_curve=results["rmse_xy_mean_curve"].numpy(),
        rmse_z_mean_curve=results["rmse_z_mean_curve"].numpy(),
        mae3d_best_curve=results["mae3d_best_curve"].numpy(),
        mae_xy_best_curve=results["mae_xy_best_curve"].numpy(),
        mae_z_best_curve=results["mae_z_best_curve"].numpy(),
        rmse3d_best_curve=results["rmse3d_best_curve"].numpy(),
        rmse_xy_best_curve=results["rmse_xy_best_curve"].numpy(),
        rmse_z_best_curve=results["rmse_z_best_curve"].numpy(),
        mae3d_cv_curve=results["mae3d_cv_curve"].numpy(),
        mae_xy_cv_curve=results["mae_xy_cv_curve"].numpy(),
        mae_z_cv_curve=results["mae_z_cv_curve"].numpy(),
        rmse3d_cv_curve=results["rmse3d_cv_curve"].numpy(),
        rmse_xy_cv_curve=results["rmse_xy_cv_curve"].numpy(),
        rmse_z_cv_curve=results["rmse_z_cv_curve"].numpy(),
        spread_mean_curve=spread_curve,
        overdispersion_index_curve=odi_curve,
        eval_indices=results["eval_indices"],
        pit_x_mean=pit_diag["pit_x_mean"],
        pit_x_std=pit_diag["pit_x_std"],
        pit_x_ks=pit_diag["pit_x_ks"],
        pit_x_std_gap=pit_diag["pit_x_std_gap"],
        pit_y_mean=pit_diag["pit_y_mean"],
        pit_y_std=pit_diag["pit_y_std"],
        pit_y_ks=pit_diag["pit_y_ks"],
        pit_y_std_gap=pit_diag["pit_y_std_gap"],
        pit_z_mean=pit_diag["pit_z_mean"],
        pit_z_std=pit_diag["pit_z_std"],
        pit_z_ks=pit_diag["pit_z_ks"],
        pit_z_std_gap=pit_diag["pit_z_std_gap"],
        intent_labels=intent_labels,
    )
    print(f"\n  Results saved to {out_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()