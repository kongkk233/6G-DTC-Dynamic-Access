#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel execution framework for IEEE TMC experiments.

Provides:
- ParallelExperimentRunner: Executes experiment runs in parallel using ProcessPoolExecutor
- RunSpec: Specification for a single experiment run
- Automatic worker count optimization based on CPU and memory
- Fallback to sequential execution on failure
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add code directory to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))


@dataclass
class RunSpec:
    """Specification for a single experiment run."""
    scenario: str
    method: str
    seed: int
    extra_params: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{self.scenario}_{self.method}_{self.seed}"


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    max_workers: Optional[int] = None  # None = auto-detect
    timeout_per_run: float = 600.0  # seconds
    fallback_to_sequential: bool = True
    memory_per_worker_gb: float = 2.0  # for auto worker calculation


def get_optimal_workers(memory_per_run_gb: float = 2.0, min_workers: int = 1) -> int:
    """
    Determine optimal worker count based on CPU and available memory.

    Args:
        memory_per_run_gb: Estimated memory per worker (default 2.0 GB)
        min_workers: Minimum number of workers

    Returns:
        Optimal number of parallel workers
    """
    cpu_count = mp.cpu_count()

    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        system_reserve_gb = 2.0
        usable_gb = max(memory_per_run_gb, available_gb - system_reserve_gb)
        memory_limited = int(usable_gb / memory_per_run_gb)
    except ImportError:
        # psutil not available, use conservative estimate
        memory_limited = max(2, cpu_count // 2)

    optimal = min(cpu_count, memory_limited)
    return max(min_workers, optimal)


def _run_single_experiment(args: Tuple) -> Dict[str, Any]:
    """
    Execute a single experiment run (called in worker process).

    Args:
        args: Tuple of (run_spec_dict, base_config, scenario_configs, model_paths, extra_cfg_overrides)

    Returns:
        Dict with run results or error
    """
    run_spec_dict, base_config, scenario_configs, model_paths, extra_cfg_overrides = args

    # Reconstruct RunSpec
    run_spec = RunSpec(**run_spec_dict)

    try:
        # Import here to ensure proper process isolation
        from main import run_once, run_constellation

        # Build config
        cfg = base_config.copy()
        if run_spec.scenario in scenario_configs:
            cfg.update(scenario_configs[run_spec.scenario])
        cfg.update(run_spec.extra_params)
        if extra_cfg_overrides:
            cfg.update(extra_cfg_overrides)
        cfg["seed"] = run_spec.seed
        cfg["show_progress"] = False

        # Apply method-specific settings
        method = run_spec.method
        if method in ("B1_3GPP", "B2_Oracle", "B3_heuristic"):
            cfg["scheduler_kind"] = "heuristic"
            cfg["nsgbs_model_path"] = None
            # B2_Oracle: no CSI delay
            if method == "B2_Oracle":
                cfg["baseline_csi_delay_ttis"] = 0
        elif method == "P1_MLP":
            cfg["scheduler_kind"] = "nsgbs"
            cfg["nsgbs_model_path"] = model_paths.get("mlp")
        elif method == "P2_ISAB":
            cfg["scheduler_kind"] = "nsgbs"
            cfg["nsgbs_model_path"] = model_paths.get("isab")

        # Run experiment - choose function based on constellation mode
        if cfg.get("enable_constellation", False):
            result = run_constellation(cfg)
        else:
            result = run_once(cfg)

        return {
            "run_id": run_spec.run_id,
            "scenario": run_spec.scenario,
            "method": run_spec.method,
            "seed": run_spec.seed,
            "success": True,
            "result": result,
        }

    except Exception as e:
        return {
            "run_id": run_spec.run_id,
            "scenario": run_spec.scenario,
            "method": run_spec.method,
            "seed": run_spec.seed,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


class ParallelExperimentRunner:
    """
    Manages parallel execution of experiment batches.

    Usage:
        runner = ParallelExperimentRunner(ParallelConfig(max_workers=4))
        results = runner.run_batch(run_specs, base_cfg, scenario_cfgs, model_paths)
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        if self.config.max_workers is None:
            self.config.max_workers = get_optimal_workers(self.config.memory_per_worker_gb)

    def run_batch(
        self,
        run_specs: List[RunSpec],
        base_config: Dict,
        scenario_configs: Dict[str, Dict],
        model_paths: Dict[str, str],
        extra_cfg_overrides: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of experiment runs in parallel.

        Args:
            run_specs: List of run specifications
            base_config: Base configuration dict
            scenario_configs: Dict mapping scenario name to config
            model_paths: Dict with 'mlp' and 'isab' model paths
            extra_cfg_overrides: Optional additional config overrides
            progress_callback: Optional callback(completed, total)

        Returns:
            List of result dicts in the same order as `run_specs`
        """
        if not run_specs:
            return []

        total = len(run_specs)
        results: List[Optional[Dict[str, Any]]] = [None] * total

        # Prepare arguments for workers (must be picklable)
        work_items = [
            (
                {
                    "scenario": spec.scenario,
                    "method": spec.method,
                    "seed": spec.seed,
                    "extra_params": spec.extra_params,
                    "run_id": spec.run_id,
                },
                base_config,
                scenario_configs,
                model_paths,
                extra_cfg_overrides,
            )
            for spec in run_specs
        ]

        try:
            # Use spawn context for clean process isolation
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=ctx,
            ) as executor:
                # Submit all runs
                futures = {
                    executor.submit(_run_single_experiment, item): i
                    for i, item in enumerate(work_items)
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout_per_run)
                        results[idx] = result
                    except Exception as e:
                        spec = run_specs[idx]
                        results[idx] = {
                            "run_id": spec.run_id,
                            "scenario": spec.scenario,
                            "method": spec.method,
                            "seed": spec.seed,
                            "success": False,
                            "error": str(e),
                        }

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        except Exception as e:
            if self.config.fallback_to_sequential:
                print(f"[ParallelRunner] Parallel execution failed ({e}), falling back to sequential")
                return self._run_sequential(
                    run_specs, base_config, scenario_configs,
                    model_paths, extra_cfg_overrides, progress_callback
                )
            raise

        # Safety: ensure no missing slots
        for i, r in enumerate(results):
            if r is None:
                spec = run_specs[i]
                results[i] = {
                    "run_id": spec.run_id,
                    "scenario": spec.scenario,
                    "method": spec.method,
                    "seed": spec.seed,
                    "success": False,
                    "error": "Missing result (unexpected worker failure)",
                }

        return results

    def _run_sequential(
        self,
        run_specs: List[RunSpec],
        base_config: Dict,
        scenario_configs: Dict[str, Dict],
        model_paths: Dict[str, str],
        extra_cfg_overrides: Optional[Dict],
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> List[Dict[str, Any]]:
        """Fallback sequential execution."""
        results = []
        total = len(run_specs)

        for i, spec in enumerate(run_specs):
            args = (
                {
                    "scenario": spec.scenario,
                    "method": spec.method,
                    "seed": spec.seed,
                    "extra_params": spec.extra_params,
                    "run_id": spec.run_id,
                },
                base_config,
                scenario_configs,
                model_paths,
                extra_cfg_overrides,
            )
            result = _run_single_experiment(args)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


def run_experiments_parallel(
    run_specs: List[RunSpec],
    base_config: Dict,
    scenario_configs: Dict[str, Dict],
    model_paths: Dict[str, str],
    max_workers: Optional[int] = None,
    extra_cfg_overrides: Optional[Dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to run experiments in parallel.

    Args:
        run_specs: List of RunSpec objects
        base_config: Base configuration dict
        scenario_configs: Dict mapping scenario name to config
        model_paths: Dict with 'mlp' and 'isab' model paths
        max_workers: Number of parallel workers (None = auto)
        extra_cfg_overrides: Optional additional config overrides
        progress_callback: Optional callback(completed, total)
        show_progress: If True and no callback, use tqdm

    Returns:
        List of result dicts
    """
    config = ParallelConfig(max_workers=max_workers)
    runner = ParallelExperimentRunner(config)

    # Setup progress bar if requested
    pbar = None
    if show_progress and progress_callback is None:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(run_specs), desc="Experiments")

            def progress_callback(completed, total):
                pbar.update(1)
        except ImportError:
            pass

    try:
        results = runner.run_batch(
            run_specs,
            base_config,
            scenario_configs,
            model_paths,
            extra_cfg_overrides,
            progress_callback,
        )
    finally:
        if pbar is not None:
            pbar.close()

    return results


def build_run_specs(
    scenarios: List[str],
    methods: List[str],
    seeds: range,
    extra_params: Optional[Dict] = None,
    prefix: str = "",
) -> List[RunSpec]:
    """
    Build list of RunSpec objects for all combinations.

    Args:
        scenarios: List of scenario names
        methods: List of method names
        seeds: Range of seeds
        extra_params: Optional extra parameters for all runs
        prefix: Prefix for run_id

    Returns:
        List of RunSpec objects
    """
    specs = []
    run_id = 0
    for scenario in scenarios:
        for method in methods:
            for seed in seeds:
                specs.append(RunSpec(
                    scenario=scenario,
                    method=method,
                    seed=seed,
                    extra_params=extra_params or {},
                    run_id=f"{prefix}{run_id}" if prefix else f"{scenario}_{method}_{seed}",
                ))
                run_id += 1
    return specs
