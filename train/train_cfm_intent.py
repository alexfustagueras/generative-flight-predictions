#!/usr/bin/env python3
"""
Train Intent-Conditioned CFM model

This script trains an IntentFlowMatchingModel that extends the original CFM
with 12 continuous + 25 one-hot intent features derived from the kinematic
profile of the 60-second input history.

Usage:
    python train_cfm_intent.py [--epochs 200] [--batch-size 2048] [--lr 3e-4]
"""

import sys
import os
import random
import argparse
import math
import time
import gc
import traceback
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

# Add this directory to path for local imports
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Add parent (notebooks/) for access to generative-flight-predictions/utils
NOTEBOOKS_DIR = SCRIPT_DIR.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

GFP_DIR = NOTEBOOKS_DIR / "generative-flight-predictions"
if str(GFP_DIR) not in sys.path:
    sys.path.insert(0, str(GFP_DIR))

from model_intent import IntentFlowMatchingModel, sample_xt_and_target, get_model_config
from intent import (
    compute_intent_features_batch,
    classify_intent_batch,
    intent_index_to_onehot,
    denormalize_history,
)

# ── Dataset with intent features ────────────────────────────────────

class IntentCFMDataset(Dataset):
    """
    Wraps X, Y, C arrays and computes intent features on-the-fly.

    Each __getitem__ returns:
        x_hist:  (T_in, 7)   normalised history
        y_fut:   (T_out, 7)  normalised future
        c_ext:   (45,)       extended context = [original_8, intent_12, onehot_25]
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        C: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        intent_features: np.ndarray,
        intent_labels: np.ndarray,
        intent_feat_mean: np.ndarray,
        intent_feat_std: np.ndarray,
    ):
        self.X = X
        self.Y = Y
        self.C = C
        self.intent_features = intent_features  # (N, 12) pre-normalised
        self.intent_labels = intent_labels      # (N,) int
        self.intent_feat_mean = intent_feat_mean
        self.intent_feat_std = intent_feat_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.Y[i], dtype=torch.float32)

        # Original 8-D context
        c_orig = torch.tensor(self.C[i], dtype=torch.float32)

        # 12-D normalised intent features
        intent_f = torch.tensor(self.intent_features[i], dtype=torch.float32)

        # 25-D one-hot intent label
        onehot = torch.zeros(25, dtype=torch.float32)
        onehot[int(self.intent_labels[i])] = 1.0

        # Concatenate: [8] + [12] + [25] = [45]
        c_ext = torch.cat([c_orig, intent_f, onehot], dim=0)

        return x, y, c_ext

def precompute_intent(
    X_norm: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    batch_size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute intent features and labels for the entire dataset.
    Works in batches to avoid memory issues with large datasets.

    Returns:
        intent_features: (N, 12)
        intent_labels: (N,) int32
    """
    N = X_norm.shape[0]
    all_feats = np.empty((N, 12), dtype=np.float32)
    all_labels = np.empty(N, dtype=np.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X_batch = np.array(X_norm[start:end])
        X_phys = denormalize_history(X_batch, feat_mean, feat_std)
        all_feats[start:end] = compute_intent_features_batch(X_phys)
        all_labels[start:end] = classify_intent_batch(X_phys)

        if (start // batch_size) % 5 == 0:
            print(f"  Intent precompute: {end}/{N} ({100*end/N:.1f}%)", flush=True)

    return all_feats, all_labels

# ── Training utilities ──────────────────────────────────────────────

class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.last_step = -1
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.last_step += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.last_step < self.warmup:
                lr = base * (self.last_step + 1) / self.warmup
            else:
                t = (self.last_step - self.warmup) / max(1, self.max_steps - self.warmup)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * t))
            g["lr"] = lr

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def swap_into(self, model):
        msd = model.state_dict()

        def _swap():
            for k, v in msd.items():
                if k not in self.shadow:
                    continue
                if v.dtype.is_floating_point:
                    tmp = v.detach().clone()
                    v.data.copy_(self.shadow[k].to(device=v.device, dtype=v.dtype))
                    self.shadow[k] = tmp

        _swap()
        return _swap

# ── Main training function ──────────────────────────────────────────

def train_intent_cfm(
    train_ds: IntentCFMDataset,
    val_ds: IntentCFMDataset,
    epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_steps: int = 2000,
    ema_decay: float = 0.9995,
    patience: int = 50,
    model_cfg: dict | None = None,
    ckpt_path: str = "best_cfm_intent.pt",
    aux_w: float = 0.05,
    accum_steps: int = 1,
    compile_mode: str = "none",
    device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = min(os.cpu_count() or 1, 8)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    cfg = model_cfg or get_model_config()
    model = IntentFlowMatchingModel(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Context dimension: {cfg.get('context_dim', 45)}")

    # Resume from checkpoint if available
    resume = False
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = OrderedDict(
            (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] Loaded {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        resume = True

    if compile_mode and str(compile_mode).lower() != "none":
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[compile] Enabled with mode={compile_mode}")
        except Exception as e:
            print(f"[compile] Skipped: {e}")
    else:
        print("[compile] Disabled")

    try:
        opt = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.95), fused=True,
        )
    except TypeError:
        opt = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
        )

    max_steps = epochs * max(1, len(train_loader))
    sched = WarmupCosine(opt, warmup_steps=warmup_steps, max_steps=max_steps, min_lr=lr * 0.05)
    ema = EMA(model, decay=ema_decay)

    pos_w, vel_w = 1.0, 0.1
    best_val, bad = float("inf"), 0
    amp_dtype = None
    if device.type == "cuda":
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        print(f"[amp] Using {'bfloat16' if bf16_ok else 'float16'} autocast")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    def run_epoch(loader, train=True, log_components=False):
        model.train() if train else model.eval()
        tot = n = 0
        pos_tot = vel_tot = aux_tot = 0.0

        if train:
            opt.zero_grad(set_to_none=True)

        for step, (xb, yb, cb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            t = torch.rand(xb.size(0), 1, device=device)

            with (
                torch.set_grad_enabled(train),
                torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
                ),
            ):
                x_t, _, eps = sample_xt_and_target(yb, t)
                v_pred = model(xb, x_t, t, cb)
                y_pred = eps + v_pred

                pos_loss = F.mse_loss(y_pred[..., :3], yb[..., :3])
                vel_loss = F.mse_loss(y_pred[..., 3:6], yb[..., 3:6])
                aux_loss = F.mse_loss(y_pred[..., 6:7], yb[..., 6:7]) if aux_w > 0 else 0.0

                loss = pos_w * pos_loss + vel_w * vel_loss + aux_w * aux_loss

                if train and accum_steps > 1:
                    loss = loss / accum_steps

            if train:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model)

            tot += float(loss) * (accum_steps if train and accum_steps > 1 else 1.0)
            n += 1
            pos_tot += float(pos_loss)
            vel_tot += float(vel_loss)
            aux_tot += float(aux_loss) if isinstance(aux_loss, torch.Tensor) else aux_loss

        avg = tot / max(1, n)
        if log_components:
            print(
                f"    Loss: total={avg:.6f} | pos={pos_tot/max(1,n):.6f} | "
                f"vel={vel_tot/max(1,n):.6f} | aux*={aux_w * aux_tot/max(1,n):.6f}"
            )
        return avg

    if resume:
        restore = ema.swap_into(model)
        best_val = run_epoch(val_loader, train=False, log_components=True)
        restore()
        print(f"[resume] Baseline val loss = {best_val:.6f}")

    print("Starting training...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True, log_components=True)

        restore = ema.swap_into(model)
        va = run_epoch(val_loader, False, log_components=True)
        restore()

        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | Train {tr:.6f} | Val {va:.6f} | lr={lr_now:.2e} | {dt:.1f}s")

        if hasattr(torch.cuda, "memory_allocated") and device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  GPU peak mem: {mem:.2f} GB")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep,
                    "train_loss": tr,
                    "val_loss": va,
                    "lr": lr_now,
                    "epoch_time_s": dt,
                })
        except ImportError:
            pass

        if va < best_val - 1e-5:
            best_val, bad = va, 0
            ema.copy_to(model)
            torch.save({"model_state": model.state_dict(), "model_cfg": cfg}, ckpt_path)
            print("  -> Saved best model")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    clean = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )
    best_model = IntentFlowMatchingModel(**ckpt["model_cfg"]).to(device)
    best_model.load_state_dict(clean, strict=True)
    return best_model

# ── Main entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Intent-Conditioned CFM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--ckpt-path", type=str, default="models/cfm_intent_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="IntentCFM")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--dataset-key", type=str, default="51397c8e8791f8ca",
                        help="Dataset cache key (directory name under dataset_cache/)")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (increase for smaller GPUs)")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="none",
        help="torch.compile mode (e.g. reduce-overhead, max-autotune, or none)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── Load dataset from cache ─────────────────────────────────────
    cache_dir = SCRIPT_DIR / "dataset_cache"
    dset_key = args.dataset_key
    files_dir = cache_dir

    print(f"Loading dataset from {files_dir}...")

    # Load norm stats
    import json
    key_info = json.loads((files_dir / f"{dset_key}.key.json").read_text())
    stats_key = key_info["stats_key"]
    norm_stats_path = files_dir / f"{stats_key}.norm_stats.json"
    if not norm_stats_path.exists():
        norm_stats_path = cache_dir / f"{stats_key}.norm_stats.json"
    norm_stats = json.loads(norm_stats_path.read_text())

    feat_mean = np.array(norm_stats["feat_mean"], dtype=np.float32)
    feat_std = np.array(norm_stats["feat_std"], dtype=np.float32)
    ctx_mean = np.array(norm_stats["ctx_mean"], dtype=np.float32)
    ctx_std = np.array(norm_stats["ctx_std"], dtype=np.float32)

    X_train = np.load(files_dir / f"{dset_key}.X_train.npy", mmap_mode="r")
    Y_train = np.load(files_dir / f"{dset_key}.Y_train.npy", mmap_mode="r")
    C_train = np.load(files_dir / f"{dset_key}.C_train.npy", mmap_mode="r")
    X_val = np.load(files_dir / f"{dset_key}.X_val.npy", mmap_mode="r")
    Y_val = np.load(files_dir / f"{dset_key}.Y_val.npy", mmap_mode="r")
    C_val = np.load(files_dir / f"{dset_key}.C_val.npy", mmap_mode="r")

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, C_train: {C_train.shape}")
    print(f"X_val:   {X_val.shape},   Y_val:   {Y_val.shape},   C_val:   {C_val.shape}")

    # ── Precompute intent features ──────────────────────────────────
    print("\nPrecomputing intent features for training set...")
    train_intent_feats, train_intent_labels = precompute_intent(X_train, feat_mean, feat_std)

    print("Precomputing intent features for validation set...")
    val_intent_feats, val_intent_labels = precompute_intent(X_val, feat_mean, feat_std)

    # Normalise intent features using training stats
    intent_mean = train_intent_feats.mean(axis=0)
    intent_std = train_intent_feats.std(axis=0) + 1e-8
    train_intent_feats_norm = ((train_intent_feats - intent_mean) / intent_std).astype(np.float32)
    val_intent_feats_norm = ((val_intent_feats - intent_mean) / intent_std).astype(np.float32)

    # Print intent distribution
    unique, counts = np.unique(train_intent_labels, return_counts=True)
    print("\nIntent distribution (training):")
    from intent import intent_name
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  {intent_name(u):30s}: {c:>8d} ({100*c/len(train_intent_labels):.1f}%)")

    # ── Build datasets ──────────────────────────────────────────────
    train_ds = IntentCFMDataset(
        X_train, Y_train, C_train,
        feat_mean, feat_std,
        train_intent_feats_norm, train_intent_labels,
        intent_mean, intent_std,
    )
    val_ds = IntentCFMDataset(
        X_val, Y_val, C_val,
        feat_mean, feat_std,
        val_intent_feats_norm, val_intent_labels,
        intent_mean, intent_std,
    )

    print(f"\nTrain dataset: {len(train_ds)} samples")
    print(f"Val dataset:   {len(val_ds)} samples")

    # ── Resolve checkpoint path ─────────────────────────────────────
    ckpt_path = SCRIPT_DIR / args.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ── W&B ─────────────────────────────────────────────────────────
    model_cfg = get_model_config()
    try:
        import wandb
        wandb_kwargs = {
            "project": args.wandb_project,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "warmup_steps": args.warmup_steps,
                "ema_decay": args.ema_decay,
                "patience": args.patience,
                "seed": args.seed,
                "context_dim": model_cfg["context_dim"],
                "accum_steps": args.accum_steps,
                "dataset_key": dset_key,
                "model": "IntentFlowMatchingModel",
                "intent_features": 12,
                "intent_classes": 25,
            },
        }
        if args.wandb_name:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_kwargs)
    except ImportError:
        print("[wandb] Not installed, skipping logging.")

    # Save intent normalisation stats alongside checkpoint
    intent_norm = {
        "intent_feat_mean": intent_mean.tolist(),
        "intent_feat_std": intent_std.tolist(),
    }
    import json
    intent_norm_path = ckpt_path.with_suffix(".intent_norm.json")
    intent_norm_path.write_text(json.dumps(intent_norm, indent=2))
    print(f"Saved intent norm stats to {intent_norm_path}")

    # ── Train ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Intent-Conditioned CFM Training")
    print("=" * 70)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Context dim: {model_cfg['context_dim']}  (8 orig + 12 intent + 25 one-hot)")
    print(f"  Checkpoint:  {ckpt_path}")
    print("=" * 70 + "\n")

    model = train_intent_cfm(
        train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        patience=args.patience,
        model_cfg=model_cfg,
        ckpt_path=str(ckpt_path),
        accum_steps=args.accum_steps,
        compile_mode=args.compile_mode,
        device=device,
    )

    print("\n" + "=" * 70)
    print(f"  Training complete! Best model saved to: {ckpt_path}")
    print("=" * 70)

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)