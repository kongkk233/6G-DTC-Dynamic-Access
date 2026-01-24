#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 6 (variant): Per-UE Average SE vs N_UE.

This script reads the standard Fig06 CSV (method, N_UE, Z, seed, avg_se, ...)
and converts system Average SE to per-UE Average SE as:

    per_ue_avg_se = avg_se / N_UE

This matches: total_throughput / (N_UE * system_bandwidth).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_utils import (
    setup_style,
    COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    ci_95,
    save_fig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 6: Per-UE Average SE vs N_UE"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig06_scalability.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig06_scalability_per_ue_se.pdf",
        help="Output figure path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively"
    )
    args = parser.parse_args()

    setup_style()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Keep consistent ordering/labels/colors
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    if not methods:
        methods = sorted(df["method"].dropna().unique().tolist())
    print(f"Methods: {methods}")

    # Use most common Z value for N_UE sweep (matches plot_fig06_scalability.py behavior)
    default_z = int(df["Z"].mode().values[0]) if len(df) > 0 and "Z" in df.columns else None
    if default_z is not None and "Z" in df.columns:
        df = df[df["Z"] == default_z].copy()

    # Compute per-UE average SE
    if "N_UE" not in df.columns or "avg_se" not in df.columns:
        print("Error: CSV must contain columns: N_UE, avg_se")
        return 1

    df = df.copy()
    df["N_UE"] = pd.to_numeric(df["N_UE"], errors="coerce")
    df["avg_se"] = pd.to_numeric(df["avg_se"], errors="coerce")
    df = df.dropna(subset=["N_UE", "avg_se"])
    df = df[df["N_UE"] > 0]
    df["per_ue_avg_se"] = df["avg_se"] / df["N_UE"]

    n_ue_values = sorted(df["N_UE"].unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    markers = {"B1_3GPP": "o", "P1_MLP": "D", "P2_ISAB": "v"}
    linestyles = {"B1_3GPP": ":", "P1_MLP": "-", "P2_ISAB": "--"}

    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) == 0:
            continue

        means = []
        cis = []
        for n_ue in n_ue_values:
            subset = method_data[method_data["N_UE"] == n_ue]
            values = subset["per_ue_avg_se"].dropna().values
            if len(values) > 0:
                means.append(float(np.mean(values)))
                cis.append(float(ci_95(values)))
            else:
                means.append(np.nan)
                cis.append(0.0)

        means = np.asarray(means, dtype=float)
        cis = np.asarray(cis, dtype=float)

        color = COLORS.get(method, "gray")
        marker = markers.get(method, "o")
        linestyle = linestyles.get(method, "-")
        label = METHOD_LABELS.get(method, method)

        ax.plot(
            n_ue_values,
            means,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=1.8,
            markersize=6,
        )
        ax.fill_between(n_ue_values, means - cis, means + cis, color=color, alpha=0.15)

    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Per-UE Average SE (bits/s/Hz per UE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=9)

    title = "Per-UE Average SE vs N_UE"
    if default_z is not None:
        title = f"{title} (Z={default_z})"
    ax.set_title(title)

    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        save_fig(fig, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

