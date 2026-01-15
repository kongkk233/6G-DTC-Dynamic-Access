#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ISAB-based NS-GBS scorer from dataset npz.
Listwise ranking + regression (optional uncertainty head).
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Create env from environment_nn.yml and retry. "
        f"Import error: {exc}"
    )


def get_default_device() -> str:
    """自动检测最佳可用设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_meta(path: Path) -> dict:
    meta_path = Path(str(path) + ".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_features(feat, actions, step, prb_count, add_z=True, add_step=True):
    x = np.asarray(feat, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Invalid feature shape: {x.shape}")
    extras = []
    if actions is not None:
        actions = np.asarray(actions, dtype=np.int64)
        if add_z and actions.shape[1] >= 3:
            denom = float(prb_count - 1) if prb_count and prb_count > 1 else 1.0
            z_norm = actions[:, 2].astype(np.float32) / denom
            extras.append(z_norm.reshape(-1, 1))
    if add_step:
        denom = float(prb_count) if prb_count and prb_count > 0 else 1.0
        step_norm = float(step) / denom
        extras.append(np.full((x.shape[0], 1), step_norm, dtype=np.float32))
    if extras:
        x = np.concatenate([x] + extras, axis=1)
    return x


def compute_norm_stats(features, actions, steps, indices, prb_count, add_z, add_step):
    sum_vec = None
    sumsq_vec = None
    count = 0
    for idx in indices:
        x = build_features(features[idx], actions[idx], steps[idx], prb_count, add_z, add_step)
        if x.size == 0:
            continue
        if sum_vec is None:
            sum_vec = x.sum(axis=0)
            sumsq_vec = (x * x).sum(axis=0)
        else:
            sum_vec += x.sum(axis=0)
            sumsq_vec += (x * x).sum(axis=0)
        count += x.shape[0]
    if count <= 0:
        raise ValueError("No features to compute normalization.")
    mean = sum_vec / float(count)
    var = sumsq_vec / float(count) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class NSGBSSetDataset(Dataset):
    def __init__(
        self,
        features,
        deltas,
        actions,
        steps,
        prb_count,
        add_z=True,
        add_step=True,
        mean=None,
        std=None,
    ):
        self.features = features
        self.deltas = deltas
        self.actions = actions
        self.steps = steps
        self.prb_count = prb_count
        self.add_z = add_z
        self.add_step = add_step
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.deltas)

    def __getitem__(self, idx):
        x = build_features(
            self.features[idx],
            self.actions[idx],
            self.steps[idx],
            self.prb_count,
            self.add_z,
            self.add_step,
        )
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        d = np.asarray(self.deltas[idx], dtype=np.float32)
        if d.ndim != 1 or d.shape[0] != x.shape[0]:
            raise ValueError(f"Delta shape mismatch at idx={idx}: {d.shape} vs {x.shape}")
        return x, d


def collate_batch(batch):
    xs, ds = zip(*batch)
    max_actions = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    bsz = len(xs)

    x_pad = np.zeros((bsz, max_actions, feat_dim), dtype=np.float32)
    d_pad = np.zeros((bsz, max_actions), dtype=np.float32)
    mask = np.zeros((bsz, max_actions), dtype=np.bool_)
    for i, (x, d) in enumerate(zip(xs, ds)):
        n = x.shape[0]
        x_pad[i, :n, :] = x
        d_pad[i, :n] = d
        mask[i, :n] = True
    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(d_pad),
        torch.from_numpy(mask),
    )


class MAB(nn.Module):
    def __init__(self, dim, num_heads, ff_hidden=256, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, k, key_mask=None):
        key_padding_mask = None if key_mask is None else ~key_mask
        out, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(q + out)
        y = self.ff(x)
        return self.ln2(x + y)


class ISAB(nn.Module):
    def __init__(self, dim, num_heads, inducing_points=32, ff_hidden=256, dropout=0.0):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, inducing_points, dim))
        self.mab1 = MAB(dim, num_heads, ff_hidden=ff_hidden, dropout=dropout)
        self.mab2 = MAB(dim, num_heads, ff_hidden=ff_hidden, dropout=dropout)

    def forward(self, x, mask=None):
        bsz = x.shape[0]
        i = self.inducing.expand(bsz, -1, -1)
        h = self.mab1(i, x, key_mask=mask)
        return self.mab2(x, h, key_mask=None)


class ISABScorer(nn.Module):
    def __init__(
        self,
        feat_dim,
        d_model=128,
        num_heads=4,
        num_layers=2,
        inducing_points=32,
        ff_hidden=256,
        dropout=0.0,
        out_dim=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                ISAB(d_model, num_heads, inducing_points, ff_hidden=ff_hidden, dropout=dropout)
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, out_dim),
        )

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, mask=mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        return self.head(h)


def masked_log_softmax(x, mask, dim=-1):
    x = x.masked_fill(~mask, -1e9)
    return F.log_softmax(x, dim=dim)


def masked_softmax(x, mask, dim=-1):
    x = x.masked_fill(~mask, -1e9)
    return F.softmax(x, dim=dim)


def listwise_loss(scores, deltas, mask, tau):
    p = masked_softmax(deltas / tau, mask)
    logq = masked_log_softmax(scores, mask)
    loss = -(p * logq).sum(dim=1)
    return loss.mean()


def regression_loss(mu, deltas, mask, use_nll=False, logvar=None):
    if use_nll:
        if logvar is None:
            raise ValueError("logvar required for NLL regression.")
        logvar = logvar.clamp(min=-10.0, max=10.0)
        var = torch.exp(logvar)
        loss = 0.5 * ((deltas - mu) ** 2 / var + logvar)
    else:
        loss = F.smooth_l1_loss(mu, deltas, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


def batch_top1_acc(scores, deltas, mask):
    scores = scores.masked_fill(~mask, -1e9)
    deltas = deltas.masked_fill(~mask, -1e9)
    pred_idx = torch.argmax(scores, dim=1)
    true_idx = torch.argmax(deltas, dim=1)
    return float((pred_idx == true_idx).float().mean().item())


def ndcg_at_k(scores, deltas, mask, k=5):
    scores = scores.detach().cpu().numpy()
    deltas = deltas.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    ndcgs = []
    for i in range(scores.shape[0]):
        valid = mask[i].astype(bool)
        if not valid.any():
            continue
        s = scores[i][valid]
        d = deltas[i][valid]
        k_eff = min(k, s.shape[0])
        order = np.argsort(-s)
        ideal = np.argsort(-d)
        rel = d - d.min()
        if np.allclose(rel, 0):
            ndcgs.append(1.0)
            continue
        denom = np.log2(np.arange(2, k_eff + 2))
        dcg = float(np.sum(rel[order[:k_eff]] / denom))
        idcg = float(np.sum(rel[ideal[:k_eff]] / denom))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train ISAB NS-GBS scorer.")
    parser.add_argument("--data", required=True, help="Path to dataset npz")
    parser.add_argument("--out", default="output/models/nsgbs_isab.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--inducing", type=int, default=32)
    parser.add_argument("--ff-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau-sweep", type=float, nargs="+", default=None,
                        help="Run tau sweep experiment with multiple values (e.g., --tau-sweep 0.1 0.2 0.3)")
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--prb-count", type=int, default=None)
    parser.add_argument("--no-add-z", action="store_true")
    parser.add_argument("--no-add-step", action="store_true")
    parser.add_argument("--add-n-ue", action="store_true", help="Add population-aware n_ue feature")
    parser.add_argument("--rbar-normalize", choices=["log", "sigmoid"], default="log", help="Rbar normalization method")
    parser.add_argument("--rbar-scale", type=float, default=1.0, help="Scale for sigmoid Rbar normalization")
    parser.add_argument("--n-ue-ref", type=float, default=100.0, help="Reference UE count for n_ue feature")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--use-nll", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--filter-gap-threshold", type=float, default=None,
                        help="Filter samples with delta gap (Top1-Top2) below this threshold")
    parser.add_argument("--save-best", action="store_true",
                        help="Save best model based on validation loss (default: save last)")
    args = parser.parse_args()

    # Handle tau sweep mode
    if args.tau_sweep:
        print(f"[Tau Sweep] Running experiments with tau = {args.tau_sweep}")
        results = []
        for tau in args.tau_sweep:
            print(f"\n{'='*60}")
            print(f"[Tau Sweep] tau = {tau}")
            print(f"{'='*60}")
            # Create a copy of args with the current tau
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.tau = tau
            sweep_args.tau_sweep = None  # Prevent recursion
            # Modify output path to include tau
            base_out = Path(args.out)
            sweep_args.out = str(base_out.parent / f"{base_out.stem}_tau{tau}{base_out.suffix}")
            # Run training
            best_val, best_acc, best_ndcg = train_model(sweep_args)
            results.append({
                "tau": tau,
                "best_val_loss": best_val,
                "best_val_acc": best_acc,
                "best_val_ndcg": best_ndcg,
                "model_path": sweep_args.out,
            })

        # Print summary
        print(f"\n{'='*60}")
        print("[Tau Sweep] Summary")
        print(f"{'='*60}")
        print(f"{'tau':>8} {'val_loss':>12} {'val_top1':>12} {'val_ndcg@5':>12}")
        print("-" * 48)
        best_result = None
        for r in results:
            print(f"{r['tau']:>8.3f} {r['best_val_loss']:>12.4f} {r['best_val_acc']:>12.3f} {r['best_val_ndcg']:>12.3f}")
            if best_result is None or r['best_val_loss'] < best_result['best_val_loss']:
                best_result = r
        if best_result:
            print(f"\nBest: tau={best_result['tau']:.3f} (val_loss={best_result['best_val_loss']:.4f})")
            print(f"Model: {best_result['model_path']}")

        # Save sweep results
        sweep_results_path = Path(args.out).parent / "tau_sweep_results.json"
        with open(sweep_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved sweep results: {sweep_results_path}")
        return

    # Normal training mode
    train_model(args)


def train_model(args) -> tuple:
    """Train model and return (best_val_loss, best_val_acc, best_val_ndcg)."""
    set_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    features = data["features"]
    deltas = data["deltas"]
    actions = data["actions"] if "actions" in data else None
    steps = data["step"] if "step" in data else None
    if actions is None or steps is None:
        raise SystemExit("Dataset missing required fields: actions/step.")
    if len(features) == 0:
        raise SystemExit("Empty dataset.")

    # Filter low-discriminability samples if threshold is set
    if args.filter_gap_threshold is not None and args.filter_gap_threshold > 0:
        keep_mask = []
        for d in deltas:
            d_arr = np.asarray(d, dtype=float)
            if d_arr.size < 2:
                keep_mask.append(False)
                continue
            sorted_d = np.sort(d_arr)[::-1]
            gap = sorted_d[0] - sorted_d[1]
            keep_mask.append(gap >= args.filter_gap_threshold)
        keep_mask = np.array(keep_mask)
        n_before = len(features)
        features = features[keep_mask]
        deltas = deltas[keep_mask]
        actions = actions[keep_mask]
        steps = steps[keep_mask]
        n_after = len(features)
        print(f"[Filter] Removed {n_before - n_after} samples with gap < {args.filter_gap_threshold:.3f} "
              f"({n_after}/{n_before} = {n_after/n_before*100:.1f}% remaining)")

    meta = load_meta(Path(args.data))
    prb_count = int(args.prb_count or meta.get("prb_count") or meta.get("config", {}).get("Z", 0) or 0)
    meta_cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    meta_add_z = bool(meta_cfg.get("nsgbs_add_z", False))
    meta_add_step = bool(meta_cfg.get("nsgbs_add_step", False))
    add_z = False if args.no_add_z else not meta_add_z
    add_step = False if args.no_add_step else not meta_add_step

    n = len(deltas)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - args.val_split))
    train_idx = idx[:split]
    val_idx = idx[split:] if split < n else idx[:0]

    mean = std = None
    if not args.no_norm:
        mean, std = compute_norm_stats(features, actions, steps, train_idx, prb_count, add_z, add_step)

    train_ds = NSGBSSetDataset(
        features[train_idx],
        deltas[train_idx],
        actions[train_idx],
        steps[train_idx],
        prb_count,
        add_z=add_z,
        add_step=add_step,
        mean=mean,
        std=std,
    )
    val_ds = (
        NSGBSSetDataset(
            features[val_idx],
            deltas[val_idx],
            actions[val_idx],
            steps[val_idx],
            prb_count,
            add_z=add_z,
            add_step=add_step,
            mean=mean,
            std=std,
        )
        if val_idx.size > 0
        else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch) if val_ds else None

    feat_dim = build_features(features[0], actions[0], steps[0], prb_count, add_z, add_step).shape[1]
    out_dim = 2 if args.uncertainty else 1
    model = ISABScorer(
        feat_dim,
        d_model=args.d_model,
        num_heads=args.heads,
        num_layers=args.layers,
        inducing_points=args.inducing,
        ff_hidden=args.ff_hidden,
        dropout=args.dropout,
        out_dim=out_dim,
    ).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = None
    best_acc = 0.0
    best_ndcg = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rank = 0.0
        total_reg = 0.0
        total_acc = 0.0
        total_cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for x, d, mask in pbar:
            x = x.to(args.device)
            d = d.to(args.device)
            mask = mask.to(args.device)

            out = model(x, mask)
            if args.uncertainty:
                mu = out[..., 0]
                logvar = out[..., 1]
            else:
                mu = out.squeeze(-1)
                logvar = None

            loss_rank = listwise_loss(mu, d, mask, args.tau)
            loss_reg = regression_loss(mu, d, mask, use_nll=args.use_nll and args.uncertainty, logvar=logvar)
            loss = loss_rank + args.lambda_reg * loss_reg

            optim.zero_grad()
            loss.backward()
            optim.step()

            bsz = x.shape[0]
            total_loss += float(loss.item()) * bsz
            total_rank += float(loss_rank.item()) * bsz
            total_reg += float(loss_reg.item()) * bsz
            batch_acc = batch_top1_acc(mu, d, mask)
            total_acc += batch_acc * bsz
            total_cnt += bsz

            # 更新进度条显示实时指标
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

        train_loss = total_loss / max(1, total_cnt)
        train_rank = total_rank / max(1, total_cnt)
        train_reg = total_reg / max(1, total_cnt)
        train_acc = total_acc / max(1, total_cnt)

        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_rank = 0.0
            v_reg = 0.0
            v_acc = 0.0
            v_cnt = 0
            v_ndcg = []
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]  ", leave=False)
                for x, d, mask in val_pbar:
                    x = x.to(args.device)
                    d = d.to(args.device)
                    mask = mask.to(args.device)
                    out = model(x, mask)
                    if args.uncertainty:
                        mu = out[..., 0]
                        logvar = out[..., 1]
                    else:
                        mu = out.squeeze(-1)
                        logvar = None
                    loss_rank = listwise_loss(mu, d, mask, args.tau)
                    loss_reg = regression_loss(mu, d, mask, use_nll=args.use_nll and args.uncertainty, logvar=logvar)
                    loss = loss_rank + args.lambda_reg * loss_reg
                    bsz = x.shape[0]
                    v_loss += float(loss.item()) * bsz
                    v_rank += float(loss_rank.item()) * bsz
                    v_reg += float(loss_reg.item()) * bsz
                    batch_acc = batch_top1_acc(mu, d, mask)
                    v_acc += batch_acc * bsz
                    v_ndcg.append(ndcg_at_k(mu, d, mask, k=5))
                    v_cnt += bsz
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")
            val_loss = v_loss / max(1, v_cnt)
            val_rank = v_rank / max(1, v_cnt)
            val_reg = v_reg / max(1, v_cnt)
            val_acc = v_acc / max(1, v_cnt)
            val_ndcg = float(np.mean(v_ndcg)) if v_ndcg else 0.0
        else:
            val_loss = None
            val_rank = None
            val_reg = None
            val_acc = None
            val_ndcg = None

        msg = (
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} rank={train_rank:.4f} reg={train_reg:.4f} top1={train_acc:.3f}"
        )
        if val_loss is not None:
            msg += (
                f" val_loss={val_loss:.4f} rank={val_rank:.4f} reg={val_reg:.4f} "
                f"top1={val_acc:.3f} ndcg@5={val_ndcg:.3f}"
            )
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_acc = val_acc
                best_ndcg = val_ndcg
                if args.save_best:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    msg += " *"
                patience_counter = 0
            else:
                patience_counter += 1
        print(msg)

        # Early stopping check
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"[Early Stop] No improvement for {patience_counter} epochs, stopping.")
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save best or last model
    if args.save_best and best_state is not None:
        torch.save(best_state, out_path)
        print(f"[NS-GBS] Saved best model (val_loss={best_val:.4f})")
    else:
        torch.save(model.state_dict(), out_path)

    model_add_z = bool(meta_add_z or add_z)
    model_add_step = bool(meta_add_step or add_step)
    model_add_n_ue = bool(args.add_n_ue)
    meta_out = {
        "model_kind": "isab",
        "feature_dim": int(feat_dim),
        "d_model": int(args.d_model),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "inducing_points": int(args.inducing),
        "ff_hidden": int(args.ff_hidden),
        "dropout": float(args.dropout),
        "tau": float(args.tau),
        "lambda_reg": float(args.lambda_reg),
        "use_uncertainty": bool(args.uncertainty),
        "use_nll": bool(args.use_nll),
        "add_z": model_add_z,
        "add_step": model_add_step,
        "add_n_ue": model_add_n_ue,
        "rbar_normalize": str(args.rbar_normalize),
        "rbar_scale": float(args.rbar_scale),
        "n_ue_ref": float(args.n_ue_ref),
        "prb_count": int(prb_count),
        "mean": None if mean is None else mean.tolist(),
        "std": None if std is None else std.tolist(),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)) if val_ds is not None else 0,
        "seed": int(args.seed),
    }
    with open(str(out_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=True, indent=2)

    print(f"[NS-GBS] saved model: {out_path}")

    return best_val if best_val is not None else 0.0, best_acc, best_ndcg


if __name__ == "__main__":
    main()
