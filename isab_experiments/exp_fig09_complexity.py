#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 9: Runtime / Complexity / Deployability.

Goal: Satisfy "practicality" review criteria by measuring compute overhead.

Collects:
- Scheduler compute time per TTI
- Model scoring overhead: score_calls, avg_score_ms_per_call

Outputs CSV with columns:
  method, device, seed, avg_se, T, score_calls, score_time_sec, ms_per_call, ms_per_tti
"""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

# Default scenario
DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"

# Default model paths
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"


def load_config_from_file(config_path: Path):
    """Load CONFIG dict from a Python config file."""
    spec = importlib.util.spec_from_file_location("scenario_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    """Load base CONFIG from code/config.py."""
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def detect_device():
    """Detect available compute device (CPU/GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def run_experiment(cfg, method, model_mlp, model_isab):
    """
    Configure and run a single experiment with timing.

    Returns:
        Dict with experiment results and timing metrics
    """
    from main import run_once

    if method == "B3_heuristic":
        cfg["scheduler_kind"] = "heuristic"
        cfg["nsgbs_model_path"] = None
    elif method == "P1_MLP":
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_mlp)
    elif method == "P2_ISAB":
        cfg["scheduler_kind"] = "nsgbs"
        cfg["nsgbs_model_path"] = str(model_isab)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Enable stats collection for NS-GBS methods
    if method in ("P1_MLP", "P2_ISAB"):
        cfg["nsgbs_collect_stats"] = True

    # Time the run
    start_time = time.perf_counter()
    result = run_once(cfg)
    end_time = time.perf_counter()

    total_time_sec = end_time - start_time
    T = cfg.get("T", 1000)

    # Extract NS-GBS stats if available
    nsgbs_stats = result.get("nsgbs_stats", {}) or {}
    score_calls = nsgbs_stats.get("score_calls", 0)
    score_time_sec = nsgbs_stats.get("score_time_sec", 0.0)
    avg_ms_per_call = nsgbs_stats.get("avg_score_ms_per_call", None)
    if avg_ms_per_call is None:
        avg_ms_per_call = (float(score_time_sec) * 1000.0) / float(score_calls) if int(score_calls) > 0 else 0.0
    avg_us_per_action = nsgbs_stats.get("avg_score_us_per_action", None)
    if avg_us_per_action is None:
        actions_total = int(nsgbs_stats.get("actions_total", 0) or 0)
        avg_us_per_action = (float(score_time_sec) * 1e6) / float(actions_total) if actions_total > 0 else 0.0

    # Compute per-TTI timing
    ms_per_tti = (total_time_sec * 1000) / T if T > 0 else 0.0
    score_ms_per_tti = (float(score_time_sec) * 1000.0) / float(T) if T > 0 else 0.0

    return {
        "avg_se": float(result.get("avg_se_radiomap", 0.0)),
        "T": T,
        "total_time_sec": float(total_time_sec),
        "score_calls": int(score_calls),
        "score_time_sec": float(score_time_sec),
        "ms_per_call": float(avg_ms_per_call),
        "us_per_action": float(avg_us_per_action),
        "ms_per_tti": float(ms_per_tti),
        "score_ms_per_tti": float(score_ms_per_tti),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Figure 9: Runtime / Complexity"
    )
    parser.add_argument(
        "--scenario-path",
        default=DEFAULT_SCENARIO_PATH,
        help=f"Path to scenario config (default: {DEFAULT_SCENARIO_PATH})"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds (default: 3, runtime is stable)"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed (default: 1)"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=500,
        help="Number of TTIs for timing (default: 500)"
    )
    parser.add_argument(
        "--N_UE",
        type=int,
        default=None,
        help="Override number of UEs"
    )
    parser.add_argument(
        "--nsgbs-max-actions",
        type=int,
        default=None,
        help="Optional: limit #candidate actions scored per step (reduces inference cost)"
    )
    parser.add_argument(
        "--methods",
        default="B3_heuristic,P1_MLP,P2_ISAB",
        help="Comma-separated methods (default: B3_heuristic,P1_MLP,P2_ISAB)"
    )
    parser.add_argument(
        "--model-mlp",
        default=DEFAULT_MLP_MODEL,
        help=f"Path to MLP model (default: {DEFAULT_MLP_MODEL})"
    )
    parser.add_argument(
        "--model-isab",
        default=DEFAULT_ISAB_MODEL,
        help=f"Path to ISAB model (default: {DEFAULT_ISAB_MODEL})"
    )
    parser.add_argument(
        "--out",
        default="output/results/fig09_complexity.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Validate model paths
    model_mlp = Path(args.model_mlp)
    model_isab = Path(args.model_isab)

    if "P1_MLP" in methods and not model_mlp.exists():
        print(f"Warning: MLP model not found: {model_mlp}")
    if "P2_ISAB" in methods and not model_isab.exists():
        print(f"Warning: ISAB model not found: {model_isab}")

    # Detect device
    device = detect_device()

    # Load configs
    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    # Calculate total runs
    total_runs = len(methods) * args.seeds

    print(f"\nFigure 9: Complexity Experiment")
    print(f"  Scenario: {args.scenario_path}")
    print(f"  Methods: {methods}")
    print(f"  Device: {device}")
    print(f"  T (TTIs): {args.T}")
    print(f"  Seeds: {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  Total runs: {total_runs}\n")

    rows = []
    pbar = tqdm(total=total_runs, desc="Fig09", disable=args.no_progress,
                bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        for method in methods:
            model_path = model_mlp if method == "P1_MLP" else (model_isab if method == "P2_ISAB" else None)
            if method in ("P1_MLP", "P2_ISAB") and model_path and not model_path.exists():
                pbar.update(args.seeds)
                continue

            for seed in range(args.seed_start, args.seed_start + args.seeds):
                cfg = base_cfg.copy()
                cfg.update(scenario_cfg)

                cfg["T"] = args.T
                if args.N_UE is not None:
                    cfg["N_UE"] = args.N_UE
                cfg["seed"] = seed
                cfg["show_progress"] = False
                # Timing runs should not write large JSON reports to disk.
                cfg["write_json_report"] = False
                # Make the measured device explicit and consistent with the reported label.
                cfg["nsgbs_device"] = device
                if args.nsgbs_max_actions is not None:
                    cfg["nsgbs_max_actions"] = int(args.nsgbs_max_actions)

                pbar.set_description(f"Fig09 [{method}/{seed}]")

                try:
                    result = run_experiment(cfg, method, model_mlp, model_isab)
                    rows.append({
                        "method": method,
                        "device": device,
                        "seed": seed,
                        "avg_se": result["avg_se"],
                        "T": result["T"],
                        "total_time_sec": result["total_time_sec"],
                        "score_calls": result["score_calls"],
                        "score_time_sec": result["score_time_sec"],
                        "ms_per_call": result["ms_per_call"],
                        "us_per_action": result["us_per_action"],
                        "ms_per_tti": result["ms_per_tti"],
                        "score_ms_per_tti": result["score_ms_per_tti"],
                    })
                    pbar.set_postfix(
                        time=f"{result['total_time_sec']:.1f}s",
                        ms_tti=f"{result['ms_per_tti']:.2f}"
                    )
                except Exception as e:
                    print(f"\nError {method}/seed={seed}: {e}")

                pbar.update(1)

    finally:
        pbar.close()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "device", "seed", "avg_se", "T", "total_time_sec",
                  "score_calls", "score_time_sec", "ms_per_call", "us_per_action",
                  "score_ms_per_tti", "ms_per_tti"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path}")
    print(f"  Total rows: {len(rows)}")

    # Print summary
    print("\n--- Timing Summary ---")
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if method_rows:
            avg_ms_tti = np.mean([r["ms_per_tti"] for r in method_rows])
            avg_inf_ms_tti = np.mean([r.get("score_ms_per_tti", 0.0) for r in method_rows])
            avg_ms_call = np.mean([r.get("ms_per_call", 0.0) for r in method_rows])
            avg_se = np.mean([r["avg_se"] for r in method_rows])
            print(
                f"  {method}: e2e={avg_ms_tti:.2f} ms/TTI, "
                f"infer={avg_inf_ms_tti:.2f} ms/TTI, "
                f"ms/call={avg_ms_call:.3f}, SE={avg_se:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
