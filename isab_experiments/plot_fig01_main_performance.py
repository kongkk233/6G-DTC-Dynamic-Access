#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot for Figure 1: Main Performance Across Scenarios.

Generates two subplots:
(a) Mean SE by method across scenarios (grouped bar with 95% CI)
(b) Gain vs B1_3GPP (%) across scenarios (grouped bar)
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
    ci_95, save_fig, add_subplot_label, format_scenario_name
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 1: Main Performance Across Scenarios"
    )
    parser.add_argument(
        "--csv",
        default="output/results/fig01_main_performance.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        default="output/figures/fig01_main_performance.pdf",
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
        print("Run exp_fig01_main_performance.py first to generate data.")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Filter to single scenarios only
    df = df[df['scenario'].str.contains('single')]

    # Get unique scenarios and methods
    scenarios = df['scenario'].unique().tolist()
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]

    # Local scenario name mapping (without "single" designation)
    def format_scenario_name_local(scenario):
        mapping = {'toronto_single': 'Toronto', 'shanghai_single': 'Shanghai'}
        return mapping.get(scenario, scenario)

    print(f"Scenarios: {scenarios}")
    print(f"Methods: {methods}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================================================================
    # Subplot (a): Mean SE by method across scenarios
    # =========================================================================
    ax = axes[0]

    x = np.arange(len(scenarios))
    bar_width = 0.18
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for i, method in enumerate(methods):
        means = []
        cis = []
        for scenario in scenarios:
            subset = df[(df['scenario'] == scenario) & (df['method'] == method)]
            se_values = subset['avg_se'].dropna().values
            if len(se_values) > 0:
                means.append(np.mean(se_values))
                cis.append(ci_95(se_values))
            else:
                means.append(0)
                cis.append(0)

        color = COLORS.get(method, f'C{i}')
        label = METHOD_LABELS.get(method, method)

        ax.bar(x + offsets[i], means, bar_width, yerr=cis,
               label=label, color=color, edgecolor='black', linewidth=0.5,
               capsize=2, error_kw={'linewidth': 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([format_scenario_name_local(s) for s in scenarios], rotation=15, ha='right')
    ax.set_ylabel('Average SE (bits/s/Hz)')
    ax.set_xlabel('Scenario')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    add_subplot_label(ax, '(a)')

    # =========================================================================
    # Subplot (b): Gain vs B1_3GPP (%) across scenarios
    # =========================================================================
    ax = axes[1]

    # Calculate gain relative to B1_3GPP for each scenario
    gain_methods = [m for m in methods if m != 'B1_3GPP']

    for i, method in enumerate(gain_methods):
        gains_mean = []
        gains_ci = []

        for scenario in scenarios:
            # Get B1_3GPP SE for this scenario (per seed)
            b1_data = df[(df['scenario'] == scenario) & (df['method'] == 'B1_3GPP')]
            method_data = df[(df['scenario'] == scenario) & (df['method'] == method)]

            if len(b1_data) > 0 and len(method_data) > 0:
                # Calculate paired gains
                b1_se = b1_data.set_index('seed')['avg_se']
                method_se = method_data.set_index('seed')['avg_se']

                # Compute gain for each paired seed
                common_seeds = b1_se.index.intersection(method_se.index)
                if len(common_seeds) > 0:
                    gains = []
                    for seed in common_seeds:
                        b1_val = b1_se.loc[seed]
                        m_val = method_se.loc[seed]
                        if b1_val > 0:
                            gain = (m_val - b1_val) / b1_val * 100
                            gains.append(gain)

                    if gains:
                        gains_mean.append(np.mean(gains))
                        gains_ci.append(ci_95(gains))
                    else:
                        gains_mean.append(0)
                        gains_ci.append(0)
                else:
                    gains_mean.append(0)
                    gains_ci.append(0)
            else:
                gains_mean.append(0)
                gains_ci.append(0)

        color = COLORS.get(method, f'C{i}')
        label = METHOD_LABELS.get(method, method)

        # Adjust offsets for fewer methods
        n_gain_methods = len(gain_methods)
        gain_offsets = np.linspace(-(n_gain_methods - 1) / 2, (n_gain_methods - 1) / 2, n_gain_methods) * bar_width

        ax.bar(x + gain_offsets[i], gains_mean, bar_width, yerr=gains_ci,
               label=label, color=color, edgecolor='black', linewidth=0.5,
               capsize=2, error_kw={'linewidth': 0.8})

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([format_scenario_name_local(s) for s in scenarios], rotation=15, ha='right')
    ax.set_ylabel('Gain vs 3GPP Baseline (%)')
    ax.set_xlabel('Scenario')
    ax.legend(loc='best', frameon=False, fontsize=8)
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
