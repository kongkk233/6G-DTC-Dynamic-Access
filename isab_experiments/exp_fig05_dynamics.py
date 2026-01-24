#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment for Figure 5: Non-Stationary Interference + Orbit Dynamics.

Goal: Validate performance under fast-changing conditions.

Supports parallel execution via --parallel flag (default: enabled).
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_DIR))

DEFAULT_SCENARIO_PATH = "test/config_toronto_single.py"
DEFAULT_MLP_MODEL = "output/models/nsgbs_scorer.pt"
DEFAULT_ISAB_MODEL = "output/models/nsgbs_isab_tau0.2.pt"
FLICKER_STD_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
DOPPLER_RESIDUAL_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def load_config_from_file(config_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("scenario_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def load_base_config():
    import importlib.util
    config_path = SCRIPT_DIR / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("base_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def main():
    parser = argparse.ArgumentParser(description="Figure 5: Dynamics")
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--N_UE", type=int, default=None)
    parser.add_argument("--flicker-values", default=",".join(map(str, FLICKER_STD_VALUES)))
    parser.add_argument("--doppler-values", default=",".join(map(str, DOPPLER_RESIDUAL_VALUES)))
    parser.add_argument("--methods", default="B1_3GPP,P1_MLP,P2_ISAB")
    parser.add_argument("--model-mlp", default=DEFAULT_MLP_MODEL)
    parser.add_argument("--model-isab", default=DEFAULT_ISAB_MODEL)
    parser.add_argument("--out", default="output/results/fig05_dynamics.csv")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    flicker_values = [float(x) for x in args.flicker_values.split(",")]
    doppler_values = [float(x) for x in args.doppler_values.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    model_mlp = SCRIPT_DIR / args.model_mlp
    model_isab = SCRIPT_DIR / args.model_isab

    base_cfg = load_base_config()
    scenario_cfg = load_config_from_file(SCRIPT_DIR / args.scenario_path)

    extra_cfg = {}
    if args.T is not None:
        extra_cfg["T"] = args.T
    if args.N_UE is not None:
        extra_cfg["N_UE"] = args.N_UE

    seed_range = range(args.seed_start, args.seed_start + args.seeds)
    total_runs = (len(flicker_values) + len(doppler_values)) * len(methods) * args.seeds

    print(f"\nFigure 5: Dynamics Experiment")
    print(f"  Flicker values: {flicker_values}")
    print(f"  Doppler values: {doppler_values}")
    print(f"  Total runs: {total_runs}")

    use_parallel = args.parallel and not args.sequential
    rows = []

    if use_parallel:
        from parallel_runner import ParallelExperimentRunner, ParallelConfig, RunSpec, get_optimal_workers

        workers = args.workers or get_optimal_workers()
        print(f"  Mode: PARALLEL ({workers} workers)\n")

        scenario_configs = {"default": scenario_cfg}
        run_specs = []

        for flicker in flicker_values:
            for method in methods:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["rm_flicker_db_std"] = flicker
                    params["doppler_residual_fraction"] = 0.0
                    params["_type"] = "flicker_db_std"
                    params["_value"] = flicker
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        for doppler in doppler_values:
            for method in methods:
                for seed in seed_range:
                    params = extra_cfg.copy()
                    params["doppler_residual_fraction"] = doppler
                    params["_type"] = "doppler_residual"
                    params["_value"] = doppler
                    run_specs.append(RunSpec(scenario="default", method=method, seed=seed, extra_params=params))

        pbar = tqdm(total=len(run_specs), desc="Fig05", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        runner = ParallelExperimentRunner(ParallelConfig(max_workers=workers))
        try:
            results = runner.run_batch(
                run_specs, base_cfg, scenario_configs,
                {"mlp": str(model_mlp), "isab": str(model_isab)},
                progress_callback=lambda c, t: (setattr(pbar, 'n', c), pbar.refresh())
            )
        finally:
            pbar.close()

        for i, r in enumerate(results):
            spec = run_specs[i]
            if r.get("success"):
                res = r["result"]
                is_baseline = r["method"] == "B1_3GPP"
                rows.append({
                    "method": r["method"],
                    "dynamic_type": spec.extra_params.get("_type", ""),
                    "dynamic_value": spec.extra_params.get("_value", 0),
                    "seed": r["seed"],
                    "avg_se": res.get("avg_se_baseline_default", 0.0) if is_baseline else res.get("avg_se_radiomap", 0.0),
                    "avg_se_baseline": res.get("avg_se_baseline_default", 0.0),
                    "gain_pct": 0.0 if is_baseline else res.get("improvement_vs_default_pct", 0.0),
                })
    else:
        print(f"  Mode: SEQUENTIAL\n")
        from main import run_once

        pbar = tqdm(total=total_runs, desc="Fig05", disable=args.no_progress,
                    bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        try:
            for flicker in flicker_values:
                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False
                        cfg["rm_flicker_db_std"] = flicker
                        cfg["doppler_residual_fraction"] = 0.0

                        if method == "B1_3GPP":
                            cfg["scheduler_kind"] = "heuristic"
                            cfg["nsgbs_model_path"] = None
                        else:
                            cfg["scheduler_kind"] = "nsgbs"
                            cfg["nsgbs_model_path"] = str(model_mlp if method == "P1_MLP" else model_isab)

                        pbar.set_description(f"Fig05 [flicker={flicker}/{method[:4]}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            is_baseline = method == "B1_3GPP"
                            rows.append({
                                "method": method, "dynamic_type": "flicker_db_std",
                                "dynamic_value": flicker, "seed": seed,
                                "avg_se": result.get("avg_se_baseline_default", 0.0) if is_baseline else result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": 0.0 if is_baseline else result.get("improvement_vs_default_pct", 0.0)
                            })
                        except Exception as e:
                            print(f"\nError flicker={flicker}/{method}/seed={seed}: {e}")
                        pbar.update(1)

            for doppler in doppler_values:
                for method in methods:
                    for seed in seed_range:
                        cfg = base_cfg.copy()
                        cfg.update(scenario_cfg)
                        cfg.update(extra_cfg)
                        cfg["seed"] = seed
                        cfg["show_progress"] = False
                        cfg["doppler_residual_fraction"] = doppler

                        if method == "B1_3GPP":
                            cfg["scheduler_kind"] = "heuristic"
                            cfg["nsgbs_model_path"] = None
                        else:
                            cfg["scheduler_kind"] = "nsgbs"
                            cfg["nsgbs_model_path"] = str(model_mlp if method == "P1_MLP" else model_isab)

                        pbar.set_description(f"Fig05 [doppler={doppler}/{method[:4]}/s{seed}]")
                        try:
                            result = run_once(cfg)
                            is_baseline = method == "B1_3GPP"
                            rows.append({
                                "method": method, "dynamic_type": "doppler_residual",
                                "dynamic_value": doppler, "seed": seed,
                                "avg_se": result.get("avg_se_baseline_default", 0.0) if is_baseline else result.get("avg_se_radiomap", 0.0),
                                "avg_se_baseline": result.get("avg_se_baseline_default", 0.0),
                                "gain_pct": 0.0 if is_baseline else result.get("improvement_vs_default_pct", 0.0)
                            })
                        except Exception as e:
                            print(f"\nError doppler={doppler}/{method}/seed={seed}: {e}")
                        pbar.update(1)
        finally:
            pbar.close()

    out_path = SCRIPT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dynamic_type", "dynamic_value", "seed", "avg_se", "avg_se_baseline", "gain_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
