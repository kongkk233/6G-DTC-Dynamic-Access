#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 6: Scalability With UE Load and Bandwidth.

Generates two subplots:
(a) SE and 5%-tile throughput vs N_UE (dual y-axis)
(b) SE vs Z (PRB count) if data available
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
    setup_style, COLORS, METHOD_LABELS, METHOD_ORDER,
    ci_95, save_fig, add_subplot_label
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 6: Scalability"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig06_scalability.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig06_scalability.pdf",
        help="Output figure path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively"
    )
    args = parser.parse_args()

    # Setup style
    setup_style()

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Get methods in order
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    print(f"Methods: {methods}")

    # Check if Z sweep data is available (multiple Z values)
    z_values = df['Z'].unique()
    has_z_sweep = len(z_values) > 1

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    markers = {'B1_3GPP': 'o', 'P1_MLP': 'D', 'P2_ISAB': 'v'}
    linestyles = {'B1_3GPP': ':', 'P1_MLP': '-', 'P2_ISAB': '--'}

    # =========================================================================
    # Subplot (a): SE vs N_UE
    # =========================================================================
    ax = axes[0]

    # Use most common Z value for N_UE sweep
    default_z = df['Z'].mode().values[0] if len(df) > 0 else 51
    df_nue = df[df['Z'] == default_z]

    n_ue_values = sorted(df_nue['N_UE'].unique())

    for method in methods:
        method_data = df_nue[df_nue['method'] == method]
        if len(method_data) == 0:
            continue

        se_means = []
        se_cis = []

        for n_ue in n_ue_values:
            subset = method_data[method_data['N_UE'] == n_ue]
            se_values = subset['avg_se'].dropna().values

            if len(se_values) > 0:
                se_means.append(np.mean(se_values))
                se_cis.append(ci_95(se_values))
            else:
                se_means.append(np.nan)
                se_cis.append(0)

        se_means = np.array(se_means)
        se_cis = np.array(se_cis)

        color = COLORS.get(method, 'gray')
        marker = markers.get(method, 'o')
        linestyle = linestyles.get(method, '-')
        label = METHOD_LABELS.get(method, method)

        ax.plot(n_ue_values, se_means, marker=marker, linestyle=linestyle, color=color,
                label=label, linewidth=1.8, markersize=6)
        ax.fill_between(n_ue_values, se_means - se_cis, se_means + se_cis, color=color, alpha=0.15)

    ax.set_xlabel('Number of UEs')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): SE vs Z (PRB count) or Gain vs N_UE
    # =========================================================================
    ax = axes[1]

    if has_z_sweep:
        # Plot SE vs Z
        default_n_ue = df['N_UE'].mode().values[0] if len(df) > 0 else 100
        df_z = df[df['N_UE'] == default_n_ue]

        for method in methods:
            method_data = df_z[df_z['method'] == method]
            if len(method_data) == 0:
                continue

            z_vals = sorted(method_data['Z'].unique())
            means = []
            cis = []

            for z in z_vals:
                subset = method_data[method_data['Z'] == z]
                se_values = subset['avg_se'].dropna().values
                if len(se_values) > 0:
                    means.append(np.mean(se_values))
                    cis.append(ci_95(se_values))
                else:
                    means.append(np.nan)
                    cis.append(0)

            means = np.array(means)
            cis = np.array(cis)
            color = COLORS.get(method, 'gray')
            marker = markers.get(method, 'o')
            linestyle = linestyles.get(method, '-')
            label = METHOD_LABELS.get(method, method)

            ax.plot(z_vals, means, marker=marker, linestyle=linestyle, color=color,
                    label=label, linewidth=1.8, markersize=6)
            ax.fill_between(z_vals, means - cis, means + cis, color=color, alpha=0.2)

        ax.set_xlabel('Number of PRBs (Z)')
        ax.set_ylabel('Average SE (bits/s/Hz)')
    else:
        # Plot Gain vs N_UE instead
        gain_methods = [m for m in methods if m != 'B1_3GPP']

        for method in gain_methods:
            method_data = df_nue[df_nue['method'] == method]
            if len(method_data) == 0:
                continue

            gains_mean = []
            gains_ci = []

            for n_ue in n_ue_values:
                subset = method_data[method_data['N_UE'] == n_ue]
                gain_values = subset['gain_pct'].dropna().values
                if len(gain_values) > 0:
                    gains_mean.append(np.mean(gain_values))
                    gains_ci.append(ci_95(gain_values))
                else:
                    gains_mean.append(np.nan)
                    gains_ci.append(0)

            gains_mean = np.array(gains_mean)
            gains_ci = np.array(gains_ci)
            color = COLORS.get(method, 'gray')
            marker = markers.get(method, 'o')
            linestyle = linestyles.get(method, '-')
            label = METHOD_LABELS.get(method, method)

            ax.plot(n_ue_values, gains_mean, marker=marker, linestyle=linestyle, color=color,
                    label=label, linewidth=1.8, markersize=6)
            ax.fill_between(n_ue_values, gains_mean - gains_ci, gains_mean + gains_ci,
                            color=color, alpha=0.2)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Number of UEs')
        ax.set_ylabel('Gain vs 3GPP Baseline (%)')

    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3)
    add_subplot_label(ax, '(b)')

    # =========================================================================
    # Finalize
    # =========================================================================
    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        save_fig(fig, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
