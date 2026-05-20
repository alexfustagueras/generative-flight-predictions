#!/usr/bin/env python3
"""
Train base (non-intent) Conditional Flow Matching (CFM) model.

This script mirrors the training settings used in notebooks/OSN_paper_training_1min.ipynb,
but is CLI-friendly for cluster runs (SLURM).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import FlowMatchingModel, get_model_config, sample_xt_and_target  # noqa: E402
from utils.utils import CFMDataset  # noqa: E402


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-6):
        self.opt = optimizer
        self.warmup = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)
        self.last_step = -1
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.last_step += 1
        for i, g in enumerate(self.opt.param_groups):
            base = float(self.base_lrs[i])
            if self.last_step < self.warmup:
                lr = base * (self.last_step + 1) / max(1, self.warmup)
            else:
                t = (self.last_step - self.warmup) / max(1, self.max_steps - self.warmup)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * t))
            g["lr"] = lr


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def swap_into(self, model: nn.Module):
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


def _make_loader(ds: CFMDataset, bs: int, shuffle: bool) -> DataLoader:
    num_workers = min(os.cpu_count() or 1, 8)
    return DataLoader(
        ds,
        batch_size=int(bs),
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def _cache_paths(cache_dir: Path, dataset_key: str) -> dict[str, Path]:
    base = cache_dir / dataset_key
    return {
        "key": base.with_suffix(".key.json"),
        "summary": base.with_suffix(".summary.json"),
        "x_tr": base.with_suffix(".X_train.npy"),
        "y_tr": base.with_suffix(".Y_train.npy"),
        "c_tr": base.with_suffix(".C_train.npy"),
        "x_va": base.with_suffix(".X_val.npy"),
        "y_va": base.with_suffix(".Y_val.npy"),
        "c_va": base.with_suffix(".C_val.npy"),
    }


def load_cached_split(cache_dir: Path, dataset_key: str):
    paths = _cache_paths(cache_dir, dataset_key)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cached dataset files:\n" + "\n".join(missing))

    key_obj = json.loads(paths["key"].read_text())
    summary_obj = json.loads(paths["summary"].read_text())

    X_train = np.load(paths["x_tr"], mmap_mode="r")
    Y_train = np.load(paths["y_tr"], mmap_mode="r")
    C_train = np.load(paths["c_tr"], mmap_mode="r")
    X_val = np.load(paths["x_va"], mmap_mode="r")
    Y_val = np.load(paths["y_va"], mmap_mode="r")
    C_val = np.load(paths["c_va"], mmap_mode="r")

    return (X_train, Y_train, C_train, X_val, Y_val, C_val, key_obj, summary_obj)


def train_base_cfm(
    train_ds: CFMDataset,
    val_ds: CFMDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    warmup_steps: int,
    ema_decay: float,
    patience: int,
    ckpt_path: Path,
    aux_w: float,
    accum_steps: int,
    compile_mode: str,
    wandb_project: str | None,
    wandb_name: str | None,
    seed: int,
) -> FlowMatchingModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = _make_loader(train_ds, batch_size, shuffle=True)
    val_loader = _make_loader(val_ds, batch_size, shuffle=False)

    cfg = get_model_config()
    model: nn.Module = FlowMatchingModel(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    resume = False
    best_val = float("inf")
    bad = 0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] Loaded {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        best_val = float(ckpt.get("best_val", best_val))
        bad = int(ckpt.get("bad_epochs", bad))
        resume = True

    if compile_mode and str(compile_mode).lower() != "none":
        try:
            model = torch.compile(model, mode=str(compile_mode))
            print(f"[compile] Enabled with mode={compile_mode}")
        except Exception as e:
            print(f"[compile] Skipped: {e}")
    else:
        print("[compile] Disabled")

    try:
        opt = optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=(0.9, 0.95),
            fused=True,
        )
    except TypeError:
        opt = optim.AdamW(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay), betas=(0.9, 0.95)
        )

    max_steps = int(epochs) * max(1, len(train_loader))
    sched = WarmupCosine(opt, warmup_steps=int(warmup_steps), max_steps=max_steps, min_lr=float(lr) * 0.05)
    ema = EMA(model, decay=float(ema_decay))

    pos_w, vel_w = 1.0, 0.1
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    wandb_run = None
    if wandb_project:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=str(wandb_project),
                name=str(wandb_name) if wandb_name else None,
                config={
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "grad_clip": float(grad_clip),
                    "warmup_steps": int(warmup_steps),
                    "ema_decay": float(ema_decay),
                    "patience": int(patience),
                    "aux_w": float(aux_w),
                    "accum_steps": int(accum_steps),
                    "compile_mode": str(compile_mode),
                    "model_cfg": cfg,
                },
            )
            print(f"[wandb] enabled: project={wandb_project}")
        except Exception as e:
            print(f"[wandb] disabled (import/init failed): {e}")
            wandb_run = None

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
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
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")),
            ):
                x_t, _, eps = sample_xt_and_target(yb, t)
                v_pred = model(xb, x_t, t, cb)
                y_pred = eps + v_pred

                pos_loss = F.mse_loss(y_pred[..., :3], yb[..., :3])
                vel_loss = F.mse_loss(y_pred[..., 3:6], yb[..., 3:6])
                aux_loss = F.mse_loss(y_pred[..., 6:7], yb[..., 6:7]) if aux_w > 0 else 0.0

                loss = pos_w * pos_loss + vel_w * vel_loss + float(aux_w) * aux_loss
                if train and int(accum_steps) > 1:
                    loss = loss / int(accum_steps)

            if train:
                loss.backward()
                if (step + 1) % int(accum_steps) == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model)

            tot += float(loss) * (int(accum_steps) if train and int(accum_steps) > 1 else 1)
            n += 1
            pos_tot += float(pos_loss)
            vel_tot += float(vel_loss)
            aux_tot += float(aux_loss) if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

        return {
            "loss": tot / max(1, n),
            "pos": pos_tot / max(1, n),
            "vel": vel_tot / max(1, n),
            "aux": aux_tot / max(1, n),
        }

    if resume:
        restore = ema.swap_into(model)
        base = run_epoch(val_loader, train=False)
        restore()
        print(f"[resume] baseline val loss={base['loss']:.6f}")

    print("Starting training…")
    for ep in range(1, int(epochs) + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, train=True)
        restore = ema.swap_into(model)
        va = run_epoch(val_loader, train=False)
        restore()
        dt = time.time() - t0

        print(
            f"Epoch {ep:03d} | "
            f"Train {tr['loss']:.6f} (pos={tr['pos']:.6f}, vel={tr['vel']:.6f}, aux={tr['aux']:.6f}) | "
            f"Val {va['loss']:.6f} (pos={va['pos']:.6f}, vel={va['vel']:.6f}, aux={va['aux']:.6f}) | "
            f"{dt:.1f}s"
        )

        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "epoch": ep,
                        "train/loss": tr["loss"],
                        "train/pos": tr["pos"],
                        "train/vel": tr["vel"],
                        "train/aux": tr["aux"],
                        "val/loss": va["loss"],
                        "val/pos": va["pos"],
                        "val/vel": va["vel"],
                        "val/aux": va["aux"],
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                )
            except Exception as e:
                print(f"[wandb] log failed: {e}")

        if va["loss"] < best_val - 1e-5:
            best_val = float(va["loss"])
            bad = 0
            ema.copy_to(model)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": cfg,
                    "best_val": best_val,
                    "bad_epochs": bad,
                    "epoch": ep,
                },
                ckpt_path,
            )
            print("  ✓ Saved best model")
        else:
            bad += 1
            if bad >= int(patience):
                print("Early stopping triggered.")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    clean = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items())
    best = FlowMatchingModel(**ckpt["model_cfg"]).to(device)
    best.load_state_dict(clean, strict=True)
    best.eval()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Train base CFM model (no intent conditioning).")
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "dataset_cache"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--ckpt-path", type=str, default=str(REPO_ROOT / "models" / "cfm_base.pt"))
    ap.add_argument("--aux-w", type=float, default=0.05)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--compile-mode", type=str, default="none")
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--wandb-name", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cache_dir = Path(args.cache_dir)
    (Xtr, Ytr, Ctr, Xva, Yva, Cva, key_obj, summary_obj) = load_cached_split(cache_dir, args.dataset_key)

    print("=== Dataset ===")
    print("dataset_key:", args.dataset_key)
    print("cache_dir:", cache_dir)
    print("prep.min_fl:", (key_obj.get("prep") or {}).get("min_fl"))
    print("prep.min_fl_mode:", (key_obj.get("prep") or {}).get("min_fl_mode"))
    print("sizes:", summary_obj.get("sizes"))
    print("windowing:", summary_obj.get("windowing"))

    train_ds = CFMDataset(Xtr, Ytr, Ctr)
    val_ds = CFMDataset(Xva, Yva, Cva)

    try:
        train_base_cfm(
            train_ds,
            val_ds,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            warmup_steps=int(args.warmup_steps),
            ema_decay=float(args.ema_decay),
            patience=int(args.patience),
            ckpt_path=Path(args.ckpt_path),
            aux_w=float(args.aux_w),
            accum_steps=int(args.accum_steps),
            compile_mode=str(args.compile_mode),
            wandb_project=(str(args.wandb_project).strip() or None),
            wandb_name=(str(args.wandb_name).strip() or None),
            seed=int(args.seed),
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

