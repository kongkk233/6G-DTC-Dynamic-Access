"""
NS-GBS torch inference utilities.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class ActionScorer(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_hidden: int = 256, dropout: float = 0.0):
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
    def __init__(self, dim: int, num_heads: int, inducing_points: int = 32, ff_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, inducing_points, dim))
        self.mab1 = MAB(dim, num_heads, ff_hidden=ff_hidden, dropout=dropout)
        self.mab2 = MAB(dim, num_heads, ff_hidden=ff_hidden, dropout=dropout)

    def forward(self, x, mask=None):
        bsz = x.shape[0]
        inducing = self.inducing.expand(bsz, -1, -1)
        h = self.mab1(inducing, x, key_mask=mask)
        return self.mab2(x, h, key_mask=None)


class ISABScorer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        inducing_points: int = 32,
        ff_hidden: int = 256,
        dropout: float = 0.0,
        out_dim: int = 1,
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


class NSGBSScorer:
    def __init__(
        self,
        model,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        device: str,
        model_kind: str = "mlp",
        use_uncertainty: bool = False,
    ):
        self.model = model
        self.device = device
        self.model_kind = str(model_kind).lower()
        self.use_uncertainty = bool(use_uncertainty)
        self.mean = None
        self.std = None
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
            self.std = torch.tensor(std, dtype=torch.float32, device=device)

    def score(self, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is not available for NS-GBS scoring.")
        x = np.asarray(features, dtype=np.float32)
        if self.model_kind == "isab":
            if x.ndim == 1:
                x = x.reshape(1, 1, -1)
            elif x.ndim == 2:
                x = x.reshape(1, x.shape[0], x.shape[1])
            elif x.ndim != 3:
                raise ValueError(f"Invalid feature shape for ISAB: {x.shape}")
            t = torch.from_numpy(x).to(self.device)
            if self.mean is not None and self.std is not None:
                t = (t - self.mean) / self.std
            # No padding is used; avoid passing an all-ones mask to reduce overhead.
            with torch.inference_mode():
                out = self.model(t, mask=None)
                if self.use_uncertainty and out.shape[-1] >= 2:
                    out = out[..., 0]
                out = out.squeeze(-1)
            return out.detach().cpu().numpy().reshape(-1)
        else:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            t = torch.from_numpy(x).to(self.device)
            if self.mean is not None and self.std is not None:
                t = (t - self.mean) / self.std
            with torch.inference_mode():
                out = self.model(t).squeeze(-1)
            return out.detach().cpu().numpy()


def _load_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nsgbs_scorer(cfg: dict) -> Optional[NSGBSScorer]:
    if torch is None:
        print("[NS-GBS] PyTorch not available; cannot load model.")
        return None
    model_path = cfg.get("nsgbs_model_path", None)
    if not model_path:
        return None
    device_cfg = cfg.get("nsgbs_device", None)
    device = str(device_cfg) if device_cfg else ("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    mean = std = None
    model_kind = "mlp"
    use_uncertainty = False

    if str(model_path).endswith(".ts"):
        model = torch.jit.load(model_path, map_location=device)
        meta_candidates = [
            str(model_path) + ".json",
            os.path.splitext(str(model_path))[0] + ".pt.json",
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)
        if meta_path is not None:
            meta = _load_meta(meta_path)
            # Expose expected feature dimension to the scheduler (for backward compatible feature sets)
            try:
                cfg["nsgbs_feature_dim"] = int(meta.get("feature_dim")) if meta.get("feature_dim") is not None else None
            except Exception:
                pass
            if "add_z" in meta:
                cfg["nsgbs_add_z"] = bool(meta.get("add_z"))
            if "add_step" in meta:
                cfg["nsgbs_add_step"] = bool(meta.get("add_step"))
            mean = np.asarray(meta.get("mean"), dtype=np.float32) if meta.get("mean") is not None else None
            std = np.asarray(meta.get("std"), dtype=np.float32) if meta.get("std") is not None else None
        else:
            print(f"[NS-GBS] TorchScript meta not found ({meta_candidates}); running without normalization.")
    else:
        meta_path = str(model_path) + ".json"
        if not os.path.exists(meta_path):
            print(f"[NS-GBS] Missing meta file: {meta_path}")
            return None
        meta = _load_meta(meta_path)
        model_kind = str(meta.get("model_kind", "mlp")).lower()
        use_uncertainty = bool(meta.get("use_uncertainty", False))
        if "add_z" in meta:
            cfg["nsgbs_add_z"] = bool(meta.get("add_z"))
        if "add_step" in meta:
            cfg["nsgbs_add_step"] = bool(meta.get("add_step"))
        feat_dim = int(meta["feature_dim"])
        # Expose expected feature dimension to the scheduler (for backward compatible feature sets)
        cfg["nsgbs_feature_dim"] = int(feat_dim)
        if model_kind == "isab":
            d_model = int(meta.get("d_model", 128))
            heads = int(meta.get("heads", 4))
            layers = int(meta.get("layers", 2))
            inducing = int(meta.get("inducing_points", 32))
            ff_hidden = int(meta.get("ff_hidden", 256))
            dropout = float(meta.get("dropout", 0.0))
            out_dim = 2 if use_uncertainty else 1
            model = ISABScorer(
                feat_dim,
                d_model=d_model,
                num_heads=heads,
                num_layers=layers,
                inducing_points=inducing,
                ff_hidden=ff_hidden,
                dropout=dropout,
                out_dim=out_dim,
            )
        else:
            hidden = int(meta.get("hidden", 128))
            depth = int(meta.get("depth", 2))
            dropout = float(meta.get("dropout", 0.0))
            model = ActionScorer(feat_dim, hidden_dim=hidden, depth=depth, dropout=dropout)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        mean = np.asarray(meta.get("mean"), dtype=np.float32) if meta.get("mean") is not None else None
        std = np.asarray(meta.get("std"), dtype=np.float32) if meta.get("std") is not None else None

    model.to(device)
    model.eval()
    return NSGBSScorer(model, mean, std, device, model_kind=model_kind, use_uncertainty=use_uncertainty)
