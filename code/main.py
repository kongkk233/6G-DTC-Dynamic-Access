# -*- coding: utf-8 -*-
"""
Radio Map–aware dynamic access simulation for direct-to-satellite downlink (DL, FDD).
- Baselines: a 3GPP-like wideband PF vs a Radio Map–aware per‑PRB/block PF.
- Radio Map R[x,y,z] (dBm) represents terrestrial interference at the UE receiver.
- Output: average spectral efficiency (bits/s/Hz), relative gain, and plots.

This repository has been simplified to DL only:
- All uplink-specific mechanics (UL open-loop PC, TA, UL Doppler pre‑comp) are removed.
- Transmit power is interpreted as DL EIRP (per‑PRB) or a total DL power budget.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
import os
import json
import time
from scipy.io import loadmat
from tqdm import tqdm
from csi import sinr_to_se_mcs, effective_sinr_eesm
from config import CONFIG
from orbit import compute_geometry_and_beam, OrbitModel, simple_beam_gain_db
from ntn_csi import snr_to_se_sched
from ntn_channel import sample_3gpp_ntn_fading
from harq import HarqManager, HarqManagerFull
from link_adapt import re_per_prb_from_config, register_mcs_tables_from_file, register_bler_curves_from_file
from constellation import ConstellationOrbit

# -----------------------
# Utility conversions
# -----------------------
def dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbm / 10.0)

def mw_to_dbm(mw: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(mw)

def thermal_noise_dbm(bw_hz: float, temp_K: float = 290.0) -> float:
    """
    Thermal noise (dBm) in bandwidth bw_hz at temperature temp_K.
    Uses kTB with -174 dBm/Hz reference at 290 K.
    """
    # If temp deviates from 290 K, adjust: -174 dBm/Hz + 10*log10(T/290)
    per_hz_dbm = -174.0 + 10.0 * np.log10(max(temp_K, 1e-9) / 290.0)
    return per_hz_dbm + 10.0 * np.log10(max(bw_hz, 1.0))

# -----------------------
# Common smoothing util
# -----------------------
def blur1d(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """
    Separable box blur of radius k (window size 2k) along a specified axis.
    Edge handling via 'edge' pad; returns array with same shape as input.
    """
    if k <= 0:
        return a
    a = np.asarray(a)
    if axis < 0:
        axis = a.ndim + axis
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (k, k)
    padded = np.pad(a, pad_width, mode='edge').cumsum(axis=axis)
    slicer_hi = [slice(None)] * a.ndim
    slicer_lo = [slice(None)] * a.ndim
    slicer_hi[axis] = slice(2 * k, None)
    slicer_lo[axis] = slice(None, -2 * k)
    window_sum = padded[tuple(slicer_hi)] - padded[tuple(slicer_lo)]
    return window_sum / float(2 * k)

# -----------------------
# Radio Map I/O
# -----------------------

def load_radio_map_from_mat(path: str,
                            var_name: str = "X_true",
                            units: str = "mW") -> np.ndarray:
    """
    Load a Radio Map from a MATLAB .mat file and return dBm tensor R[x,y,z].
    Supports both traditional MAT files and HDF5-based MAT files (v7.3).
    - units: one of {"mW", "W", "dBm"}
    - var_name: variable name inside MAT file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Radio Map file not found: {path}")
    
    # Check if file is HDF5 format (MATLAB v7.3)
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            # HDF5 format - MATLAB v7.3
            if var_name in f:
                X = np.array(f[var_name], dtype=float)
            else:
                # Find first dataset if var_name not found
                keys = list(f.keys())
                if not keys:
                    raise KeyError(f"No data variables found in {path}")
                pick_key = keys[0]
                print(f"[load_radio_map_from_mat] '{var_name}' not found. Using '{pick_key}' instead.")
                X = np.array(f[pick_key], dtype=float)
    except (OSError, ImportError):
        # Traditional MAT format - try scipy.io.loadmat
        try:
            data = loadmat(path)
            # MATLAB loader brings meta keys like __header__/__version__/__globals__
            keys = [k for k in data.keys() if not k.startswith("__")]
            pick_key = var_name if var_name in data else (keys[0] if keys else None)
            if pick_key is None:
                raise KeyError(f"No data variables found in {path}. Raw keys: {list(data.keys())}")
            if var_name not in data:
                print(f"[load_radio_map_from_mat] '{var_name}' not found. Using '{pick_key}' instead.")
            X = np.array(data[pick_key], dtype=float)
        except Exception as e:
            raise ValueError(f"Failed to load MAT file {path}: {e}")
    
    # Convert units to dBm
    if units.lower() == "dbm":
        R_dbm = X
    elif units.lower() == "mw":
        R_dbm = 10.0 * np.log10(np.maximum(X, 1e-30))
    elif units.lower() == "w":
        R_dbm = 10.0 * np.log10(np.maximum(X * 1e3, 1e-30))
    else:
        raise ValueError(f"Unsupported units: {units} (use 'mW', 'W', or 'dBm')")
    return R_dbm
 

 

# -----------------------
# Helper blocks (refactor run_once)
# -----------------------
def se_from_snr(snr_lin: np.ndarray, use_mcs: bool, mcs_params: Optional[Dict] = None) -> np.ndarray:
    """
    Strategy function: map SNR (linear) to SE (bits/s/Hz).
    - If use_mcs: use an MCS mapping (approximate table for now).
    - Else: Shannon log2(1+SNR).
    mcs_params reserved for future (BLER targets, code rates, etc.).
    """
    # Apply optional frequency-offset induced ICI penalty (first-order approximation)
    if mcs_params is not None:
        eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
        if eps_f > 0.0:
            scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
            T_sym = 1.0 / (scs_khz * 1e3)
            ici_factor = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            snr_lin = np.asarray(snr_lin) / ici_factor
    if use_mcs:
        sinr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
        if mcs_params is not None:
            sinr_db = sinr_db + float(mcs_params.get("olla_offset_db", 0.0))
            table = mcs_params.get("mcs_table", "legacy")
        else:
            table = "legacy"
        return sinr_to_se_mcs(sinr_db, table=table)
    return np.log2(1.0 + np.maximum(snr_lin, 0.0))

def se_from_snr_with_split(snr_base: Union[float, np.ndarray], k_prb: int, use_mcs: bool,
                           mcs_params: Optional[Dict] = None) -> np.ndarray:
    """Apply power-split (if k_prb>1) and map via strategy."""
    k = max(1, int(k_prb))
    return se_from_snr(np.asarray(snr_base) / float(k), use_mcs, mcs_params)

def se_from_cap_shannon_with_split(cap_se: Union[float, np.ndarray], k_prb: int) -> np.ndarray:
    """
    Fallback when only Shannon SE is available (no SNR): adjust for power split.
    gamma = (2^SE - 1)/k; SE' = log2(1+gamma)
    """
    k = max(1, int(k_prb))
    gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_se)) - 1.0) / float(k)
    return np.log2(1.0 + gamma)

def _apply_ici_penalty_lin(snr_lin: np.ndarray, mcs_params: Optional[Dict]) -> np.ndarray:
    """Apply residual CFO/ICI penalty on linear SNR if configured in mcs_params."""
    if mcs_params is None:
        return np.asarray(snr_lin)
    eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
    if eps_f <= 0.0:
        return np.asarray(snr_lin)
    scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
    T_sym = 1.0 / (scs_khz * 1e3)
    ici_factor = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
    return np.asarray(snr_lin) / ici_factor

def _block_se_from_snr_vec(
    snr_lin_vec: np.ndarray,
    k_prb: int,
    use_mcs: bool,
    mcs_params: Optional[Dict],
    eesm_beta_db: float,
) -> float:
    """
    Compute per-PRB SE for a contiguous RB block using power split over k_prb PRBs.
    - If use_mcs: apply ICI penalty, divide SNR by k, map vector to effective SINR via EESM, then to SE via MCS.
    - Else (Shannon): average log2(1+SNR/k) across the block.
    Returns per-PRB SE (not the block sum).
    """
    k = max(1, int(k_prb))
    s = np.asarray(snr_lin_vec, dtype=float)
    s = _apply_ici_penalty_lin(s, mcs_params)
    if use_mcs:
        sinr_db_vec = 10.0 * np.log10(np.maximum(s / float(k), 1e-12))
        sinr_eff_db = effective_sinr_eesm(sinr_db_vec, beta_db=float(eesm_beta_db), axis=-1)
        # Apply OLLA offset if present
        if mcs_params is not None:
            sinr_eff_db = sinr_eff_db + float(mcs_params.get("olla_offset_db", 0.0))
        se = float(np.asarray(sinr_to_se_mcs(sinr_eff_db, table=mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy")))
        return se
    else:
        se_vec = np.log2(1.0 + np.maximum(s / float(k), 0.0))
        return float(np.mean(se_vec))

def se_metric_strategy(use_mcs: bool,
              snr_lin: Optional[np.ndarray] = None,
              cap_shannon: Optional[np.ndarray] = None,
              mcs_params: Optional[Dict] = None) -> np.ndarray:
    """
    Strategy for scheduling metric SE:
    - prefer mapping from snr_lin if provided;
    - otherwise, if not using MCS and Shannon SE is provided, return it.
    """
    if snr_lin is not None:
        return se_from_snr(snr_lin, use_mcs, mcs_params)
    if (not use_mcs) and (cap_shannon is not None):
        return cap_shannon
    raise ValueError("se_metric_strategy requires snr_lin or (cap_shannon with use_mcs=False)")

def select_radio_map(config: Dict, use_cache: bool = True) -> Tuple[np.ndarray, int, int, int]:
    """
    Load Radio Map from a MATLAB .mat file. The map's Z dimension must match the PRB count.
    Returns (R_xyz_dbm, X, Y, Z).

    Args:
        config: Configuration dict with radio_map_mat_path, radio_map_mat_var, radio_map_units
        use_cache: If True, use RadioMapCache to avoid repeated disk I/O (default: True)
    """
    expected_Z = int(config["Z"]) if ("Z" in config and config["Z"] is not None) else None
    path = config.get("radio_map_mat_path")
    if not path:
        raise ValueError("radio_map_mat_path must be provided (MAT file with 3D Radio Map)")

    var_name = config.get("radio_map_mat_var", "X_true")
    units = config.get("radio_map_units", "mW")

    if use_cache:
        try:
            from resource_cache import RadioMapCache
            cache = RadioMapCache.get_instance()
            R_xyz_dbm = cache.get_or_load(path, var_name, units, load_radio_map_from_mat)
        except ImportError:
            # Fallback if cache module not available
            R_xyz_dbm = load_radio_map_from_mat(path, var_name=var_name, units=units)
    else:
        R_xyz_dbm = load_radio_map_from_mat(path, var_name=var_name, units=units)

    if R_xyz_dbm.ndim != 3:
        raise ValueError(f"Loaded Radio Map must be 3D, got shape {R_xyz_dbm.shape}")
    X, Y, Z = R_xyz_dbm.shape
    # Verify Z matches expected PRB count
    if expected_Z is not None and Z != expected_Z:
        raise ValueError(f"Radio Map Z dimension ({Z}) does not match configured PRB count ({expected_Z}). "
                         f"Please provide a map with Z={expected_Z} or update config['Z'] to {Z}.")
    return R_xyz_dbm, X, Y, Z

def generate_ue_positions(N_UE: int, X: int, Y: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random UE grid indices of shape [N_UE, 2]."""
    return np.stack([rng.integers(0, X, size=N_UE), rng.integers(0, Y, size=N_UE)], axis=1)

 

def resolve_noise_and_prb_bw(config: Dict) -> Tuple[float, Optional[float]]:
    """
    Resolve thermal noise level (dBm) from SCS -> PRB BW; require SCS.
    """
    if "scs_khz" not in config or config["scs_khz"] is None:
        raise ValueError("scs_khz must be set to compute PRB bandwidth for noise")
    prb_bw_hz = float(config["scs_khz"]) * 1e3 * 12.0
    noise_dbm = thermal_noise_dbm(prb_bw_hz, temp_K=config.get("noise_temp_K", 290.0))
    return noise_dbm, prb_bw_hz

def apply_open_loop_power_control(config: Dict,
                                  L_fs_per_ue: Union[np.ndarray, float],
                                  G_rx_per_ue: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    DL-only simplification: return configured DL per‑PRB EIRP `P_tx_dbm` as-is.
    This function remains for interface compatibility.
    """
    return config["P_tx_dbm"]

def compute_metric_override_static_if_needed(config: Dict,
                                             R_xyz_dbm: np.ndarray,
                                             ue_pos: np.ndarray,
                                             L_fs_per_ue,
                                             G_rx_per_ue,
                                             noise_dbm: float,
                                             elev_deg_per_ue: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    If estimation error/blur configured, compute a static predicted per-PRB SE metric to
    override instantaneous metric in RadioMap scheduler (matches original behavior).
    """
    if config.get("radiomap_est_error_db", 0.0) <= 0.0 and config.get("radiomap_blur_sigma", 0.0) <= 0.0:
        return None
    rng = np.random.default_rng(config["seed"])
    R_hat_dbm = R_xyz_dbm.copy()
    err_db = rng.normal(0.0, config.get("radiomap_est_error_db", 0.0), size=R_hat_dbm.shape)
    R_hat_dbm = R_hat_dbm + err_db
    sig = config.get("radiomap_blur_sigma", 0.0)
    if sig and sig > 0.0:
        k = int(max(1, round(sig)))
        if k > 1:
            R_hat_dbm = blur1d(R_hat_dbm, k, axis=0)
            R_hat_dbm = blur1d(R_hat_dbm, k, axis=1)
    cap_pred, _, _, _, snr_lin_pred, _ = compute_caps(
        R_hat_dbm, ue_pos,
        P_tx_dbm=config["P_tx_dbm"],  # keep original semantics (no PC here)
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=config["shadow_std_db"],
        N0_dbm=noise_dbm,
        rx_nf_db=config.get("rx_nf_db", 0.0),
        impl_loss_db=config.get("impl_loss_db", 0.0),
        seed=config["seed"],
        elevation_deg=elev_deg_per_ue,
        channel_model=config.get("channel_model", "3gpp_ntn"),
        channel_params=config.get("channel_params"),
        channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
    )
    mcs_params = {
        "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
        "mcs_table": config.get("csi_mcs_table", "legacy"),
        "residual_freq_hz": config.get("residual_freq_hz", 0.0),
        "scs_khz": config.get("scs_khz", 30),
    }
    return se_from_snr(snr_lin_pred, config.get("use_mcs", False), mcs_params=mcs_params) if snr_lin_pred is not None else cap_pred

def build_time_variation_if_enabled(config: Dict,
                                    R_xyz_dbm: np.ndarray,
                                    ue_pos: np.ndarray,
                                    L_fs_per_ue,
                                    G_rx_per_ue,
                                    elev_deg_per_ue,
                                    P_tx_per_ue_dbm,
                                    noise_dbm: float,
                                    rng: np.random.Generator,
                                    metric_override: Optional[np.ndarray],
                                    orbit_model: Optional[object] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    If time variation is enabled, build time series of per-PRB and wideband metrics
    (both SE and SNR). Mirrors the original behavior including optional prediction
    under estimation error/blur.
    Returns a dict with keys: 'se_time_rm', 'se_time_wb', 'snr_time', 'snr_wb_time'.
    """
    if not config.get("enable_time_varying", False):
        return None
    T = config["T"]
    N_UE = int(ue_pos.shape[0])
    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))
    se_time_rm: list = []
    se_time_wb: list = []
    snr_time: list = []
    snr_wb_time: list = []
    tau_time: list = []
    fd_time: list = []
    # Keep a drifted base map and apply flicker as a stationary (per-TTI) perturbation.
    # IMPORTANT: do not accumulate flicker over time (random walk), which would make
    # interference variance grow with t and artificially inflate SE.
    R_base_t = R_xyz_dbm.copy()
    if orbit_model is None:
        orbit_model = OrbitModel(config, R_xyz_dbm.shape[0], R_xyz_dbm.shape[1]) if config.get("enable_orbit_dynamics", False) else None

    # DL-only: UL-specific pre-compensation and TA models removed.
    for t in range(T):
        if t > 0 and (vx or vy):
            R_base_t = np.roll(R_base_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
        if flicker > 0.0:
            # Interference flicker is modeled as a *wideband* (PRB-correlated) perturbation by default.
            # Rationale: per-PRB i.i.d. flicker can create artificial frequency diversity that an
            # oracle-per-PRB scheduler exploits, leading to SE increasing with "flicker std".
            kind = str(config.get("rm_flicker_kind", "global")).lower().strip()
            dist = str(config.get("rm_flicker_dist", "rectified")).lower().strip()
            if dist in ("gaussian", "normal", "signed"):
                def sample_delta(shape):
                    return rng.normal(0.0, flicker, size=shape)
            elif dist in ("rectified", "relu", "positive"):
                def sample_delta(shape):
                    return np.maximum(0.0, rng.normal(0.0, flicker, size=shape))
            elif dist in ("abs", "absolute", "half_normal"):
                def sample_delta(shape):
                    return np.abs(rng.normal(0.0, flicker, size=shape))
            else:
                raise ValueError(f"Unknown rm_flicker_dist '{dist}'.")
            if kind in ("global", "scalar"):
                R_t = R_base_t + float(sample_delta(None))
            elif kind in ("pixel", "per_pixel", "xy", "wideband_xy"):
                delta = sample_delta((R_base_t.shape[0], R_base_t.shape[1], 1))
                R_t = R_base_t + delta
            elif kind in ("prb", "per_prb", "z"):
                delta = sample_delta((1, 1, R_base_t.shape[2]))
                R_t = R_base_t + delta
            elif kind in ("element", "per_element", "xyz"):
                R_t = R_base_t + sample_delta(R_base_t.shape)
            else:
                raise ValueError(f"Unknown rm_flicker_kind '{kind}'.")
        else:
            R_t = R_base_t

        # Predictive schedule metric under imperfect map (optional, matches original gating)
        cap_pred_t = cap_wb_pred_t = snr_lin_pred_t = snr_lin_wb_pred_t = None
        if (metric_override is None) and (config.get("radiomap_est_error_db", 0.0) > 0.0 or config.get("radiomap_blur_sigma", 0.0) > 0.0):
            R_hat_dbm = R_t.copy()
            err_db = rng.normal(0.0, config.get("radiomap_est_error_db", 0.0), size=R_hat_dbm.shape)
            R_hat_dbm = R_hat_dbm + err_db
            sig = config.get("radiomap_blur_sigma", 0.0)
            if sig and sig > 0.0:
                k = int(max(1, round(sig)))
                if k > 1:
                    R_hat_dbm = blur1d(R_hat_dbm, k, axis=0)
                    R_hat_dbm = blur1d(R_hat_dbm, k, axis=1)
            if orbit_model is not None:
                L_fs_hat, G_rx_hat, _, _, elev_hat = orbit_model.get_geometry(ue_pos, t)
            else:
                L_fs_hat, G_rx_hat, elev_hat = L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue
            cap_pred_t, cap_wb_pred_t, _, _, snr_lin_pred_t, snr_lin_wb_pred_t = compute_caps(
                R_hat_dbm, ue_pos,
                P_tx_dbm=P_tx_per_ue_dbm,
                L_fs_db=L_fs_hat,
                G_rx_db=G_rx_hat,
                shadow_db_std=config["shadow_std_db"],
                N0_dbm=noise_dbm,
                rx_nf_db=config.get("rx_nf_db", 0.0),
                impl_loss_db=config.get("impl_loss_db", 0.0),
                seed=config["seed"],
                elevation_deg=elev_hat,
                channel_model=config.get("channel_model", "3gpp_ntn"),
                channel_params=config.get("channel_params"),
                channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
            )

        if orbit_model is not None:
            L_fs_t, G_rx_t, tau_s_t, fd_hz_t, elev_t = orbit_model.get_geometry(ue_pos, t)
            tau_time.append(tau_s_t)
            fd_time.append(fd_hz_t)
        else:
            L_fs_t, G_rx_t, elev_t = L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue
        cap_t, cap_wb_t, _, _, snr_lin_t, snr_lin_wb_t = compute_caps(
            R_t, ue_pos,
            P_tx_dbm=P_tx_per_ue_dbm,
            L_fs_db=L_fs_t,
            G_rx_db=G_rx_t,
            shadow_db_std=config["shadow_std_db"],
            N0_dbm=noise_dbm,
            rx_nf_db=config.get("rx_nf_db", 0.0),
            impl_loss_db=config.get("impl_loss_db", 0.0),
            seed=config["seed"],
            elevation_deg=elev_t,
            channel_model=config.get("channel_model", "3gpp_ntn"),
            channel_params=config.get("channel_params"),
            channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
        )
        # DL-only: generic residual Doppler fraction -> ICI penalty (optional)
        dop_frac = float(config.get("doppler_residual_fraction", 0.0) or 0.0)
        if (orbit_model is not None) and (dop_frac > 0.0):
            eps_f = np.abs(fd_hz_t) * dop_frac
            scs_khz = float(config.get("scs_khz", 30))
            T_sym = 1.0 / (scs_khz * 1e3)
            ici_base = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            ici_fac = np.maximum(1.0, ici_base * (1.0 + 2.0 * dop_frac))
            snr_lin_t = snr_lin_t / ici_fac.reshape(-1, 1)
            snr_lin_wb_t = snr_lin_wb_t / ici_fac
            if snr_lin_pred_t is not None:
                snr_lin_pred_t = snr_lin_pred_t / ici_fac.reshape(-1, 1)
            if snr_lin_wb_pred_t is not None:
                snr_lin_wb_pred_t = snr_lin_wb_pred_t / ici_fac
        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }
        if snr_lin_pred_t is not None:
            se_time_rm.append(se_from_snr(snr_lin_pred_t, config.get("use_mcs", False), mcs_params=mcs_params))
            se_time_wb.append(se_from_snr(snr_lin_wb_pred_t, config.get("use_mcs", False), mcs_params=mcs_params))
        else:
            se_time_rm.append(se_from_snr(snr_lin_t, config.get("use_mcs", False), mcs_params=mcs_params))
            se_time_wb.append(se_from_snr(snr_lin_wb_t, config.get("use_mcs", False), mcs_params=mcs_params))
        snr_time.append(snr_lin_t)
        snr_wb_time.append(snr_lin_wb_t)

    return {
        "se_time_rm": np.stack(se_time_rm, axis=0),      # [T, UE, Z]
        "se_time_wb": np.stack(se_time_wb, axis=0),      # [T, UE]
        "snr_time": np.stack(snr_time, axis=0),          # [T, UE, Z]
        "snr_wb_time": np.stack(snr_wb_time, axis=0),    # [T, UE]
        "tau_time": None if len(tau_time) == 0 else np.stack(tau_time, axis=0),   # [T, UE]
        "fd_time": None if len(fd_time) == 0 else np.stack(fd_time, axis=0),      # [T, UE]
    }
def compute_caps(R_xyz_dbm: np.ndarray,
                 ue_pos_xy: np.ndarray,
                 P_tx_dbm: Union[np.ndarray, float],
                 L_fs_db: Union[np.ndarray, float],
                 G_rx_db: Union[np.ndarray, float],
                 shadow_db_std: float = 5.0,
                 N0_dbm: float = -121.45,
                 rx_nf_db: float = 0.0,
                 impl_loss_db: float = 0.0,
                 seed: int = 1,
                 elevation_deg: Optional[np.ndarray] = None,
                 channel_model: str = "3gpp_ntn",
                 channel_params: Optional[Dict] = None,
                 channel_profile: str = "s_band_handheld_urban") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-UE per-PRB spectral efficiency based on the Radio Map.
    Returns:
      cap_shannon[UE,Z], cap_wb_shannon[UE], P_rx_dbm[UE], I_total_dbm[UE,Z],
      snr_lin[UE,Z], snr_lin_wb[UE]
    channel_model:
      - '3gpp_ntn': three-state NTN fading (default)
      - 'lognormal': legacy independent lognormal fading with Shannon capacity
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = R_xyz_dbm.shape
    N_UE = ue_pos_xy.shape[0]

    # UE positions
    x_idx = ue_pos_xy[:, 0]
    y_idx = ue_pos_xy[:, 1]

    # Broadcast-compatible operations for scalar or per-UE arrays
    L_fs = np.asarray(L_fs_db, dtype=float)
    G_rx = np.asarray(G_rx_db, dtype=float)
    P_tx = np.asarray(P_tx_dbm, dtype=float)
    if L_fs.ndim == 0:
        L_fs = np.full(N_UE, float(L_fs))
    if G_rx.ndim == 0:
        G_rx = np.full(N_UE, float(G_rx))
    if P_tx.ndim == 0:
        P_tx = np.full(N_UE, float(P_tx))
    channel_kind = (channel_model or "3gpp_ntn").lower().strip()
    if elevation_deg is None:
        elev_use = np.full(N_UE, 90.0, dtype=float)
    else:
        elev_use = np.asarray(elevation_deg, dtype=float)
        if elev_use.ndim == 0:
            elev_use = np.full(N_UE, float(elev_use))
    overrides = channel_params if isinstance(channel_params, dict) else None

    if channel_kind == "3gpp_ntn":
        large_scale_db, fading_lin, _ = sample_3gpp_ntn_fading(
            rng,
            elev_use,
            Z,
            profile_name=str(channel_profile or "s_band_handheld_urban"),
            overrides=overrides,
        )
        P_rx_dbm = P_tx - L_fs + G_rx + large_scale_db
    elif channel_kind in ("lognormal", "legacy"):
        shadow_db = rng.normal(0.0, shadow_db_std, size=N_UE)
        fading_lin = np.ones((N_UE, Z), dtype=float)
        P_rx_dbm = P_tx - L_fs + G_rx + shadow_db
    else:
        raise ValueError(f"Unsupported channel_model '{channel_model}'.")

    P_rx_prb_dbm = P_rx_dbm.reshape(-1, 1) + 10.0 * np.log10(np.maximum(fading_lin, 1e-12))

    # Interference + thermal noise per UE per PRB (dBm)
    I_uez_dbm = R_xyz_dbm[x_idx, y_idx, :]  # [UE,Z]
    # Effective thermal noise incl. receiver NF and implementation loss (modeled as noise rise)
    N0_eff_dbm = N0_dbm + rx_nf_db + impl_loss_db
    I_total_mw = dbm_to_mw(I_uez_dbm) + dbm_to_mw(N0_eff_dbm)
    I_total_dbm = mw_to_dbm(I_total_mw)

    # Per-PRB SNR and capacity (bits/s/Hz)
    gamma_db = P_rx_prb_dbm - I_total_dbm                          # [UE,Z]
    snr_lin = 10.0 ** (gamma_db / 10.0)
    cap = np.log2(1.0 + snr_lin)                                  # [UE,Z]

    # Wideband (3GPP-like) interference. Use mean across subbands for a more
    # conservative and realistic CQI statistic vs median.
    I_wb_mw = np.mean(I_total_mw, axis=1)                          # [UE]
    I_wb_dbm = mw_to_dbm(I_wb_mw)
    P_rx_mw = dbm_to_mw(P_rx_dbm)
    mean_fading = np.mean(fading_lin, axis=1)
    snr_lin_wb = (P_rx_mw * mean_fading) / np.maximum(I_wb_mw, 1e-30)
    snr_lin_wb = np.maximum(snr_lin_wb, 1e-12)
    cap_wb = np.log2(1.0 + snr_lin_wb)                            # [UE]

    return cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb

def pf_schedule_baseline(cap_wb: np.ndarray,
                         Z: int,
                         T: int,
                         beta: float = 0.1,
                         snr_lin_wb: np.ndarray = None,
                         overhead_eff: float = 1.0,
                         use_mcs: bool = False,
                         power_split: bool = False,
                         mcs_params: Optional[Dict] = None,
                         se_metric_time: Optional[np.ndarray] = None,
                         snr_lin_wb_time: Optional[np.ndarray] = None,
                         snr_lin_prb: Optional[np.ndarray] = None,
                         cap_prb: Optional[np.ndarray] = None,
                         snr_lin_time_prb: Optional[np.ndarray] = None,
                         force_wideband_throughput: bool = False,
                         ue_mask_time: Optional[np.ndarray] = None,
                         harq_mgr: Optional[HarqManager] = None,
                         config: Optional[Dict] = None) -> float:
    """
    3GPP-like baseline: proportional fair with wideband CQI (same cap on every PRB).
    To avoid one-UE monopolization, assign PRBs in each TTI across the top sqrt(N) UEs 
    per PF metric, equally split.
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    cfg = config or {}
    harq_priority_bonus = float(cfg.get("harq_retx_priority_bonus", 0.0))
    re_per_prb_val: Optional[int] = None
    scheduler_kind = str(cfg.get("scheduler_kind", "heuristic")).lower()
    use_nsgbs = (scheduler_kind == "nsgbs")
    nsgbs_scorer = None
    if use_nsgbs:
        try:
            from nsgbs import load_nsgbs_scorer
            # Use cache if available and enabled
            use_model_cache = cfg.get("use_model_cache", True)
            if use_model_cache:
                try:
                    from resource_cache import ModelCache
                    model_path = cfg.get("nsgbs_model_path")
                    device = cfg.get("nsgbs_device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
                    cache = ModelCache.get_instance()
                    nsgbs_scorer = cache.get_or_load(model_path, device, load_nsgbs_scorer, cfg)
                except ImportError:
                    nsgbs_scorer = load_nsgbs_scorer(cfg)
            else:
                nsgbs_scorer = load_nsgbs_scorer(cfg)
            if nsgbs_scorer is None:
                print("[NS-GBS] No model loaded; falling back to heuristic scoring.")
                use_nsgbs = False
        except Exception as exc:
            print(f"[NS-GBS] Failed to load model ({exc}); falling back to heuristic scoring.")
            use_nsgbs = False
    collect_dataset = bool(cfg.get("nsgbs_collect_dataset", False))
    dataset_out = cfg.get("nsgbs_dataset_out") if collect_dataset else None
    if collect_dataset and dataset_out is None:
        dataset_out = []
        cfg["nsgbs_dataset_out"] = dataset_out
    dataset_stride = max(1, int(cfg.get("nsgbs_collect_stride", 1) or 1))
    dataset_max_samples = cfg.get("nsgbs_collect_max_samples", None)
    dataset_topb = max(1, int(cfg.get("nsgbs_topB", 4) or 4))
    dataset_window = max(1, int(cfg.get("nsgbs_window", 3) or 3))
    if dataset_window % 2 == 0:
        dataset_window += 1
    dataset_use_harq = bool(cfg.get("nsgbs_use_harq_features", True))
    add_z_feat = bool(cfg.get("nsgbs_add_z", False))
    add_step_feat = bool(cfg.get("nsgbs_add_step", False))
    dataset_enabled = collect_dataset and (dataset_out is not None)
    dataset_count = 0
    N_UE = cap_wb.shape[0]
    # Use Shannon cap as metric by default; if use_mcs, convert to MCS SE (k=1) for metric
    if se_metric_time is None:
        metric_se = se_metric_strategy(use_mcs, snr_lin=snr_lin_wb, cap_shannon=cap_wb, mcs_params=mcs_params)
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0
    
    # Progress bar for TTI loop (only show if enabled in config)
    show_progress = bool(cfg.get("show_progress", True))
    leave_bar = bool(cfg.get("progress_leave", True))
    pbar_iter = tqdm(range(T), desc="Baseline", unit="TTI", disable=not show_progress,
                     leave=leave_bar,
                     bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for t_idx in pbar_iter:
        # Apply HARQ feedback and credit goodput if a full HARQ manager is used
        if harq_mgr is not None:
            ack_bits = None
            try:
                # Incremental time advancement to support both T-steps and repeated T=1 calls
                ack_bits = harq_mgr.advance_time()
            except TypeError:
                ack_bits = None
            if ack_bits is not None:
                if re_per_prb_val is None:
                    re_per_prb_val = max(1, re_per_prb_from_config(cfg))
                thr_ack = np.asarray(ack_bits, dtype=float) / float(re_per_prb_val)
                sum_rate += float(np.sum(thr_ack))
                Rbar = (1 - beta) * Rbar + beta * thr_ack
        if se_metric_time is not None:
            metric_se_t = se_metric_time[t_idx]
        else:
            metric_se_t = metric_se
        metric = metric_se_t / Rbar
        # Apply UE mask if provided (mask out by setting to very negative)
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)
            blocked = ~mask_t
            if np.any(blocked):
                metric = np.array(metric, copy=True)
                metric[blocked] = -1e9
        # Apply HARQ gating (cannot schedule if all processes occupied)
        if harq_mgr is not None:
            if 'mask_t' in locals():
                mask_h = np.array([harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool)
                metric[~mask_h] = -1e9
            else:
                mask_h = np.array([harq_mgr.can_schedule(u) for u in range(N_UE)], dtype=bool)
                metric = np.array(metric, copy=True)
                metric[~mask_h] = -1e9
        U_select = min(N_UE, max(3, int(np.sqrt(N_UE))))
        if U_select < N_UE:
            top_idx = np.argpartition(-metric, U_select - 1)[:U_select]
            top_idx = top_idx[np.argsort(-metric[top_idx])]
            selected = top_idx
        else:
            selected = np.argsort(-metric)
        # If mask provided, filter selected by allowed UEs
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)
            sel = [u for u in selected if mask_t[u]]
            if len(sel) == 0:
                sel = [selected[0]]  # fallback to avoid empty
            selected = np.array(sel, dtype=int)
            U_select = len(selected)

        # Determine number of PRBs per selected UE
        alloc_counts = np.full(U_select, Z // U_select, dtype=int)
        remainder = Z - int(alloc_counts.sum())
        if remainder > 0:
            alloc_counts[:remainder] += 1

        # Build per-PRB winners by round-robin across selected set (no subband awareness)
        winners = np.full(Z, -1, dtype=int)
        rem = alloc_counts.copy()
        k_ptr = 0
        for z in range(Z):
            # find next selected UE with remaining quota
            for _ in range(U_select):
                if rem[k_ptr] > 0:
                    winners[z] = selected[k_ptr]
                    rem[k_ptr] -= 1
                    k_ptr = (k_ptr + 1) % U_select
                    break
                k_ptr = (k_ptr + 1) % U_select
            if winners[z] < 0:
                winners[z] = selected[0]

        thr_i = np.zeros(N_UE)
        # Count PRBs per UE for power split
        if power_split:
            counts = np.bincount(winners, minlength=N_UE)
        else:
            counts = np.ones(N_UE, dtype=int)
        for z in range(Z):
            ue = winners[z]
            k_prb = int(counts[ue]) if power_split else 1
            # Strict wideband-only throughput (ignore PRB-level info)
            if force_wideband_throughput:
                if (snr_lin_wb is not None) or (snr_lin_wb_time is not None):
                    snr_base = snr_lin_wb_time[t_idx, ue] if snr_lin_wb_time is not None else snr_lin_wb[ue]
                    se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
                else:
                    se = se_from_cap_shannon_with_split(metric_se_t[ue], k_prb if power_split else 1)
            else:
                # Prefer per-PRB SNR/SE if available for fairness
                if (snr_lin_prb is not None) or (snr_lin_time_prb is not None):
                    if snr_lin_time_prb is not None:
                        snr_base = snr_lin_time_prb[t_idx, ue, z]
                    else:
                        snr_base = snr_lin_prb[ue, z]
                    se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
                elif cap_prb is not None and (not use_mcs):
                    se = se_from_cap_shannon_with_split(cap_prb[ue, z], k_prb if power_split else 1)
                else:
                    # Fallback to wideband
                    if (snr_lin_wb is not None) or (snr_lin_wb_time is not None):
                        snr_base = snr_lin_wb_time[t_idx, ue] if snr_lin_wb_time is not None else snr_lin_wb[ue]
                        se = se_from_snr_with_split(snr_base, k_prb if power_split else 1, use_mcs, mcs_params=mcs_params)
                    else:
                        se = se_from_cap_shannon_with_split(metric_se_t[ue], k_prb if power_split else 1)
            thr_i[ue] += se * overhead_eff
        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i
        # Update HARQ after scheduling: mark scheduled UEs (unique) for minimal manager
        if harq_mgr is not None and hasattr(harq_mgr, 'on_scheduled'):
            harq_mgr.on_scheduled(np.unique(winners))

    # Tail flush: realize ACKs that arrive after the last scheduled TTI (T-1)
    if harq_mgr is not None and bool(cfg.get("harq_flush_tail", True)):
        try:
            D = int(cfg.get("harq_ack_delay_ttis", 0) or 0)
        except Exception:
            D = 0
        if D > 0:
            if re_per_prb_val is None:
                re_per_prb_val = max(1, re_per_prb_from_config(cfg))
            for s in range(1, D + 1):
                try:
                    # Incremental tail advancement
                    ack_bits_tail = harq_mgr.advance_time()
                except TypeError:
                    ack_bits_tail = None
                if ack_bits_tail is not None:
                    thr_ack = np.asarray(ack_bits_tail, dtype=float) / float(re_per_prb_val)
                    sum_rate += float(np.sum(thr_ack))
                    Rbar = (1 - beta) * Rbar + beta * thr_ack

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

# Removed legacy per-PRB RM PF in favor of contiguous-block scheduler


def pf_schedule_radiomap_blocks(
    cap: np.ndarray,
    T: int,
    beta: float,
    snr_lin: Optional[np.ndarray],
    overhead_eff: float,
    use_mcs: bool,
    power_split: bool,
    se_metric_override: Optional[np.ndarray],
    max_prbs_per_ue: Optional[int],
    mcs_params: Optional[Dict],
    se_metric_time: Optional[np.ndarray],
    snr_lin_time: Optional[np.ndarray],
    eesm_beta_db: float = 1.0,
    require_contiguous: bool = True,
    rng: Optional[np.random.Generator] = None,
    ue_mask_time: Optional[np.ndarray] = None,
    harq_mgr: Optional[HarqManager] = None,
    # DL power allocation
    dl_power_model: str = "equal_prb",
    P_tot_dbm: Optional[float] = None,
    P_ref_dbm: Optional[float] = None,
    p_min_dbm: Optional[float] = None,
    p_max_dbm: Optional[float] = None,
    # Optional recording of per-TTI PRB assignments (winners per PRB)
    record_assignments: bool = False,
    assignments_out: Optional[list] = None,
    # Optional per-TTI per-UE throughput (SE sum across assigned PRBs)
    record_ue_thr: bool = False,
    ue_thr_out: Optional[list] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    Enhanced Radio Map–aware PF with contiguous RB blocks (single-MCS via EESM),
    power-aware greedy allocation (marginal ΔSE with power split), and robust/exploration.
    Returns average sum spectral efficiency per PRB (bits/s/Hz).
    """
    cfg = config or {}
    harq_priority_bonus = float(cfg.get("harq_retx_priority_bonus", 0.0))
    re_per_prb_val: Optional[int] = None
    scheduler_kind = str(cfg.get("scheduler_kind", "heuristic")).lower()
    use_nsgbs = (scheduler_kind == "nsgbs")
    nsgbs_scorer = None
    if use_nsgbs:
        try:
            from nsgbs import load_nsgbs_scorer
            # Use cache if available and enabled
            use_model_cache = cfg.get("use_model_cache", True)
            if use_model_cache:
                try:
                    from resource_cache import ModelCache
                    model_path = cfg.get("nsgbs_model_path")
                    device = cfg.get("nsgbs_device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
                    cache = ModelCache.get_instance()
                    nsgbs_scorer = cache.get_or_load(model_path, device, load_nsgbs_scorer, cfg)
                except ImportError:
                    nsgbs_scorer = load_nsgbs_scorer(cfg)
            else:
                nsgbs_scorer = load_nsgbs_scorer(cfg)
            if nsgbs_scorer is None:
                print("[NS-GBS] No model loaded; falling back to heuristic scoring.")
                use_nsgbs = False
        except Exception as exc:
            print(f"[NS-GBS] Failed to load model ({exc}); falling back to heuristic scoring.")
            use_nsgbs = False
    collect_dataset = bool(cfg.get("nsgbs_collect_dataset", False))
    dataset_out = cfg.get("nsgbs_dataset_out") if collect_dataset else None
    if collect_dataset and dataset_out is None:
        dataset_out = []
        cfg["nsgbs_dataset_out"] = dataset_out
    dataset_stride = max(1, int(cfg.get("nsgbs_collect_stride", 1) or 1))
    dataset_max_samples = cfg.get("nsgbs_collect_max_samples", None)
    dataset_topb = max(1, int(cfg.get("nsgbs_topB", 4) or 4))
    dataset_window = max(1, int(cfg.get("nsgbs_window", 3) or 3))
    if dataset_window % 2 == 0:
        dataset_window += 1
    dataset_use_harq = bool(cfg.get("nsgbs_use_harq_features", True))
    add_z_feat = bool(cfg.get("nsgbs_add_z", False))
    add_step_feat = bool(cfg.get("nsgbs_add_step", False))
    dataset_enabled = collect_dataset and (dataset_out is not None)
    dataset_count = 0
    collect_stats = bool(cfg.get("nsgbs_collect_stats", False))
    stats_out = cfg.get("nsgbs_stats_out") if collect_stats else None
    if collect_stats and stats_out is None:
        stats_out = {}
        cfg["nsgbs_stats_out"] = stats_out
    stats = {"steps": 0, "actions_total": 0, "score_calls": 0, "score_time_sec": 0.0} if collect_stats else None
    nsgbs_score_error_printed = False
    nsgbs_max_actions = cfg.get("nsgbs_max_actions", None)
    try:
        nsgbs_max_actions = None if nsgbs_max_actions is None else int(nsgbs_max_actions)
        if nsgbs_max_actions is not None and nsgbs_max_actions <= 0:
            nsgbs_max_actions = None
    except Exception:
        nsgbs_max_actions = None

    N_UE, Z = cap.shape
    rng = np.random.default_rng(0) if rng is None else rng

    def to_robust_sinr_db(arr_snr_lin: np.ndarray) -> np.ndarray:
        # Minimal: no uncertainty subtraction; plain SINR(dB)
        sinr_db = 10.0 * np.log10(np.maximum(arr_snr_lin, 1e-12))
        return sinr_db

    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0

    # Progress bar for TTI loop
    show_progress = bool(cfg.get("show_progress", True))
    leave_bar = bool(cfg.get("progress_leave", True))
    pbar_iter = tqdm(range(T), desc="RadioMap", unit="TTI", disable=not show_progress,
                     leave=leave_bar,
                     bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for t_idx in pbar_iter:
        if harq_mgr is not None:
            ack_bits = None
            try:
                # Incremental time advancement to support both T-steps and repeated T=1 calls
                ack_bits = harq_mgr.advance_time()
            except TypeError:
                ack_bits = None
            if ack_bits is not None:
                if re_per_prb_val is None:
                    re_per_prb_val = max(1, re_per_prb_from_config(cfg))
                thr_ack = np.asarray(ack_bits, dtype=float) / float(re_per_prb_val)
                sum_rate += float(np.sum(thr_ack))
                Rbar = (1 - beta) * Rbar + beta * thr_ack
        mask_t = None
        if ue_mask_time is not None:
            mask_t = np.asarray(ue_mask_time[t_idx], dtype=bool)
        # Prepare predicted per-PRB seed scores (k=1) and robust inputs
        if se_metric_time is not None:
            # Use given per-PRB SE predictions directly as seed scores
            se_pred_k1 = np.asarray(se_metric_time[t_idx], dtype=float)  # [UE,Z]
            sinr_db_pred = None
        else:
            # Derive from snr_lin (preferred) or from cap via Shannon inversion
            if snr_lin_time is not None:
                snr_mat = np.asarray(snr_lin_time[t_idx], dtype=float)
            elif snr_lin is not None:
                snr_mat = np.asarray(snr_lin, dtype=float)
            else:
                # Invert Shannon to get SNR from cap
                gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap)) - 1.0)
                snr_mat = gamma
            # Apply robust offset in dB if requested when building seed SE
            sinr_db_pred = to_robust_sinr_db(snr_mat)
            # For seeds we use per-PRB k=1
            if use_mcs:
                table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
                se_pred_k1 = sinr_to_se_mcs(sinr_db_pred, table=table)
            else:
                se_pred_k1 = np.log2(1.0 + snr_mat)

        # If override provided, prefer it for the scheduler metric (P3: imperfect map)
        if se_metric_time is None and (se_metric_override is not None):
            se_pred_k1 = np.asarray(se_metric_override, dtype=float)

        snr_true_t = None
        if dataset_enabled:
            if snr_lin_time is not None:
                snr_true_t = np.asarray(snr_lin_time[t_idx], dtype=float)
            elif snr_lin is not None:
                snr_true_t = np.asarray(snr_lin, dtype=float)
            else:
                gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap)) - 1.0)
                snr_true_t = gamma

        # Winners and blocks
        winners = np.full(Z, -1, dtype=int)
        l_idx = np.full(N_UE, -1, dtype=int)
        r_idx = np.full(N_UE, -1, dtype=int)
        k_assigned = np.zeros(N_UE, dtype=int)
        block_se_pred = np.zeros(N_UE, dtype=float)  # per-PRB SE of current block (predicted, for scheduling)

        # Precompute PRB order per UE for fast seeding
        order_per_ue = np.argsort(-se_pred_k1, axis=1)
        ptr_per_ue = np.zeros(N_UE, dtype=int)

        def next_unassigned_best(ue: int) -> Optional[int]:
            """
            Return the best (highest predicted SE) PRB index for this UE that is still unassigned.
            Important: do not advance past an unassigned PRB just by *enumerating* actions; only
            skip PRBs that are already assigned (they will never become free again).
            """
            ptr = int(ptr_per_ue[ue])
            ord_row = order_per_ue[ue]
            while ptr < ord_row.size and winners[int(ord_row[ptr])] >= 0:
                ptr += 1
            ptr_per_ue[ue] = ptr  # persistently skip only already-assigned PRBs
            if ptr < ord_row.size:
                return int(ord_row[ptr])
            return None

        def can_grow_left(ue: int) -> bool:
            return l_idx[ue] > 0 and winners[l_idx[ue] - 1] < 0

        def can_grow_right(ue: int) -> bool:
            return r_idx[ue] >= 0 and r_idx[ue] < (Z - 1) and winners[r_idx[ue] + 1] < 0

        def pred_block_se(ue: int, li: int, ri: int) -> float:
            """Predicted per-PRB SE for UE over [li..ri] used for scheduling metric."""
            if se_metric_time is None and (se_metric_override is None) and (sinr_db_pred is not None):
                # Use robust SINR + EESM if MCS; else Shannon mean
                snr_lin_vec = 10.0 ** (sinr_db_pred[ue, li:ri + 1] / 10.0)
                return _block_se_from_snr_vec(snr_lin_vec, ri - li + 1 if power_split else 1, use_mcs, mcs_params, eesm_beta_db)
            elif se_metric_time is not None:
                return float(np.mean(se_metric_time[t_idx, ue, li:ri + 1]))
            else:
                # Fall back to override mean (k=1 assumption)
                return float(np.mean(se_pred_k1[ue, li:ri + 1]))

        def apply_assign(ue: int, z: int) -> None:
            nonlocal winners, l_idx, r_idx, k_assigned, block_se_pred
            winners[z] = ue
            if k_assigned[ue] == 0:
                l_idx[ue] = r_idx[ue] = z
            else:
                if z == l_idx[ue] - 1:
                    l_idx[ue] = z
                elif z == r_idx[ue] + 1:
                    r_idx[ue] = z
                else:
                    # Non-contiguous assignment; if required, do nothing (should not happen)
                    l_idx[ue] = min(l_idx[ue], z) if l_idx[ue] >= 0 else z
                    r_idx[ue] = max(r_idx[ue], z) if r_idx[ue] >= 0 else z
            k_assigned[ue] += 1
            # Update predicted block SE for scheduling metric
            li, ri = int(l_idx[ue]), int(r_idx[ue])
            block_se_pred[ue] = pred_block_se(ue, li, ri)

        def retx_bonus_for_ue(ue: int) -> float:
            if (harq_mgr is not None) and hasattr(harq_mgr, 'get_retx_ues'):
                try:
                    _retx_mask = np.asarray(harq_mgr.get_retx_ues(), dtype=bool)
                    if _retx_mask[ue]:
                        return harq_priority_bonus
                except Exception:
                    return 0.0
            return 0.0

        def iter_actions():
            """Yield candidate actions in the same order as the legacy heuristic scan."""
            for ue in range(N_UE):
                if (harq_mgr is not None) and (not harq_mgr.can_schedule(ue)):
                    continue
                if (mask_t is not None) and (not mask_t[ue]):
                    continue
                # Respect per-UE PRB cap
                if (max_prbs_per_ue is not None) and (k_assigned[ue] >= int(max_prbs_per_ue)):
                    continue
                k0 = int(k_assigned[ue])
                if k0 == 0:
                    if use_nsgbs and dataset_topb > 1:
                        ord_row = order_per_ue[ue]
                        picked = 0
                        for z in ord_row:
                            zz = int(z)
                            if winners[zz] < 0:
                                yield ("seed", ue, zz)
                                picked += 1
                                if picked >= dataset_topb:
                                    break
                    else:
                        z0 = next_unassigned_best(ue)
                        if z0 is not None:
                            yield ("seed", ue, int(z0))
                    continue
                # Grow actions (contiguous if required)
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                if (not require_contiguous) or can_grow_left(ue):
                    zl = li - 1 if k0 > 0 else None
                    if zl is not None and zl >= 0 and winners[zl] < 0:
                        yield ("grow_left", ue, int(zl))
                if (not require_contiguous) or can_grow_right(ue):
                    zr = ri + 1 if k0 > 0 else None
                    if zr is not None and zr < Z and winners[zr] < 0:
                        yield ("grow_right", ue, int(zr))

        def score_action_heuristic(action) -> float:
            kind, ue, z = action
            if kind == "seed":
                se_new = float(se_pred_k1[ue, z])
                delta = se_new  # block sum gain for first PRB
            else:
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                se_old = float(block_se_pred[ue])
                if kind == "grow_left":
                    li_new, ri_new = int(z), ri
                else:
                    li_new, ri_new = li, int(z)
                se_new = pred_block_se(ue, li_new, ri_new)
                k0 = int(k_assigned[ue])
                k_new = k0 + 1
                delta = k_new * se_new - k0 * se_old
            metric = delta / Rbar[ue]
            metric += retx_bonus_for_ue(ue)
            return metric

        def score_action_nsgbs(action) -> float:
            if nsgbs_scorer is None:
                return score_action_heuristic(action)
            feat = build_nsgbs_features(action)
            score = nsgbs_scorer.score(feat)
            if isinstance(score, np.ndarray):
                return float(score.reshape(-1)[0])
            return float(score)

        def score_action(action) -> float:
            return score_action_nsgbs(action) if use_nsgbs else score_action_heuristic(action)

        kind_to_id = {"seed": 0, "grow_left": 1, "grow_right": 2, "fallback": 3}

        def window_vals(vec: np.ndarray, center: int, win: int) -> np.ndarray:
            radius = win // 2
            out = np.empty(win, dtype=float)
            for i in range(-radius, radius + 1):
                idx = center + i
                if idx < 0:
                    idx = 0
                elif idx >= vec.size:
                    idx = vec.size - 1
                out[i + radius] = vec[idx]
            return out

        def pred_delta_for_action(action) -> float:
            kind, ue, z = action
            k0 = int(k_assigned[ue])
            if k0 == 0 or kind == "seed":
                se_new = float(se_pred_k1[ue, z])
                return se_new
            li, ri = int(l_idx[ue]), int(r_idx[ue])
            se_old = float(block_se_pred[ue])
            if kind == "grow_left":
                li_new, ri_new = int(z), ri
            elif kind == "grow_right":
                li_new, ri_new = li, int(z)
            else:
                li_new, ri_new = li, ri
            se_new = pred_block_se(ue, li_new, ri_new)
            k_new = k0 + 1 if kind in ("grow_left", "grow_right") else k0
            return k_new * se_new - k0 * se_old

        def build_nsgbs_features(action) -> np.ndarray:
            kind, ue, z = action
            se_vec = np.asarray(se_pred_k1[ue], dtype=float)
            win = window_vals(se_vec, int(z), dataset_window)
            k0 = int(k_assigned[ue])
            block_pred = float(block_se_pred[ue]) if k0 > 0 else 0.0
            expected_dim = int(cfg.get("nsgbs_feature_dim", 0) or 0)
            # Backward compatible feature set selection:
            # - legacy MLP models were trained with an explicit delta_pred and PF metric (delta/Rbar)
            # - newer models (e.g., ISAB) use a log-compressed PF feature and omit delta_pred
            delta_pred = float(pred_delta_for_action(action))
            rbar = float(Rbar[ue])
            retx_flag = 1.0 if (dataset_use_harq and retx_bonus_for_ue(ue) > 0.0) else 0.0
            if max_prbs_per_ue is None:
                cap_rem = -1.0
            else:
                cap_rem = float(int(max_prbs_per_ue) - k0)
            kind_id = float(kind_to_id.get(kind, 3))
            extras = []
            if add_z_feat:
                denom = float(Z - 1) if Z > 1 else 1.0
                extras.append(np.array([float(z) / denom], dtype=float))
            if add_step_feat:
                denom = float(Z) if Z > 0 else 1.0
                extras.append(np.array([float(assigned_cnt) / denom], dtype=float))
            extras_dim = len(extras)
            legacy_dim = int(dataset_window) + 7 + extras_dim
            use_legacy = (expected_dim == legacy_dim) if expected_dim else False
            if use_legacy:
                pf_metric = delta_pred / (rbar + 1e-6)
                tail = np.array([k0, block_pred, delta_pred, pf_metric, retx_flag, cap_rem, kind_id], dtype=float)
            else:
                pf_log = -np.log(rbar + 1e-6)
                tail = np.array([k0, block_pred, pf_log, retx_flag, cap_rem, kind_id], dtype=float)
            if extras:
                feat = np.concatenate([win, tail] + extras)
            else:
                feat = np.concatenate([win, tail])
            if expected_dim and feat.size != expected_dim:
                raise ValueError(f"NS-GBS feature_dim mismatch: built {feat.size}, expected {expected_dim}")
            return feat

        def delta_true_for_action(action, snr_true: np.ndarray) -> float:
            kind, ue, z = action
            k0 = int(k_assigned[ue])
            if k0 == 0 or kind in ("seed", "fallback"):
                snr_vec = snr_true[ue, int(z):int(z) + 1]
                se_new = _block_se_from_snr_vec(
                    snr_vec,
                    1,
                    use_mcs,
                    mcs_params,
                    eesm_beta_db,
                )
                return se_new
            li, ri = int(l_idx[ue]), int(r_idx[ue])
            snr_old = snr_true[ue, li:ri + 1]
            se_old = _block_se_from_snr_vec(
                snr_old,
                k0 if power_split else 1,
                use_mcs,
                mcs_params,
                eesm_beta_db,
            )
            if kind == "grow_left":
                li_new, ri_new = int(z), ri
            elif kind == "grow_right":
                li_new, ri_new = li, int(z)
            else:
                li_new, ri_new = li, ri
            k_new = k0 + 1 if kind in ("grow_left", "grow_right") else k0
            snr_new = snr_true[ue, li_new:ri_new + 1]
            se_new = _block_se_from_snr_vec(
                snr_new,
                k_new if power_split else 1,
                use_mcs,
                mcs_params,
                eesm_beta_db,
            )
            return k_new * se_new - k0 * se_old

        def iter_actions_for_dataset(seed_topb: int):
            for ue in range(N_UE):
                if (harq_mgr is not None) and (not harq_mgr.can_schedule(ue)):
                    continue
                if (mask_t is not None) and (not mask_t[ue]):
                    continue
                if (max_prbs_per_ue is not None) and (k_assigned[ue] >= int(max_prbs_per_ue)):
                    continue
                k0 = int(k_assigned[ue])
                if k0 == 0:
                    ord_row = order_per_ue[ue]
                    picked = 0
                    for z in ord_row:
                        if winners[int(z)] < 0:
                            yield ("seed", ue, int(z))
                            picked += 1
                            if picked >= seed_topb:
                                break
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                if (not require_contiguous) or can_grow_left(ue):
                    zl = li - 1 if k0 > 0 else None
                    if zl is not None and zl >= 0 and winners[zl] < 0:
                        yield ("grow_left", ue, int(zl))
                if (not require_contiguous) or can_grow_right(ue):
                    zr = ri + 1 if k0 > 0 else None
                    if zr is not None and zr < Z and winners[zr] < 0:
                        yield ("grow_right", ue, int(zr))

        # If HARQ has pending retransmissions with resource constraints, pre-assign their blocks
        if (harq_mgr is not None) and hasattr(harq_mgr, 'get_retx_requirements'):
            try:
                reqs = harq_mgr.get_retx_requirements()
                for ue_req, (li0, ri0) in reqs.items():
                    li0 = max(0, int(li0)); ri0 = min(Z-1, int(ri0))
                    # Skip if UE masked or UE can't be scheduled
                    if (mask_t is not None) and (not mask_t[ue_req]):
                        continue
                    if not harq_mgr.can_schedule(ue_req):
                        continue
                    # Assign PRBs if free
                    can_assign = True
                    for zz in range(li0, ri0+1):
                        if winners[zz] >= 0:
                            can_assign = False
                            break
                    if not can_assign:
                        continue
                    for zz in range(li0, ri0+1):
                        winners[zz] = int(ue_req)
                    l_idx[ue_req] = li0
                    r_idx[ue_req] = ri0
                    k_assigned[ue_req] = (ri0 - li0 + 1)
                    block_se_pred[ue_req] = pred_block_se(int(ue_req), li0, ri0)
            except Exception:
                pass

        # Greedy allocation until all PRBs assigned
        assigned_cnt = 0
        # Count already assigned by pre-assignment
        assigned_cnt = int(np.sum(winners >= 0))
        while assigned_cnt < Z:
            if collect_stats and use_nsgbs and (nsgbs_scorer is not None):
                stats["steps"] += 1
            if dataset_enabled and (assigned_cnt % dataset_stride == 0):
                actions_ds = list(iter_actions_for_dataset(dataset_topb))
                if actions_ds and (snr_true_t is not None):
                    feats = [build_nsgbs_features(a) for a in actions_ds]
                    deltas = np.array([delta_true_for_action(a, snr_true_t) for a in actions_ds], dtype=np.float32)
                    label = int(np.argmax(deltas))
                    actions_arr = np.array(
                        [(kind_to_id.get(a[0], 3), int(a[1]), int(a[2])) for a in actions_ds],
                        dtype=np.int16,
                    )
                    dataset_out.append({
                        "features": np.asarray(feats, dtype=np.float32),
                        "label": label,
                        "actions": actions_arr,
                        "deltas": deltas,
                        "t_idx": int(t_idx),
                        "step": int(assigned_cnt),
                    })
                    dataset_count += 1
                    if (dataset_max_samples is not None) and (dataset_count >= int(dataset_max_samples)):
                        dataset_enabled = False

            # Build best action per UE: seed or grow L/R
            best_action = None  # (kind, ue, z_to_assign)
            actions = None
            if use_nsgbs and (nsgbs_scorer is not None):
                actions = list(iter_actions())
                # Optional pruning to reduce model inference cost: keep only top-K actions
                # using the heuristic PF metric as a cheap filter.
                if actions and (nsgbs_max_actions is not None) and (len(actions) > nsgbs_max_actions):
                    try:
                        prune_scores = np.asarray([score_action_heuristic(a) for a in actions], dtype=float)
                        k = int(nsgbs_max_actions)
                        idx = np.argpartition(prune_scores, -k)[-k:]
                        idx = idx[np.argsort(prune_scores[idx])[::-1]]
                        actions = [actions[int(j)] for j in idx]
                    except Exception:
                        actions = actions[: int(nsgbs_max_actions)]
                if collect_stats:
                    stats["actions_total"] += len(actions)
                if actions:
                    try:
                        feats = np.asarray([build_nsgbs_features(a) for a in actions], dtype=np.float32)
                        t0 = time.perf_counter()
                        scores = np.asarray(nsgbs_scorer.score(feats), dtype=float).reshape(-1)
                        dt = time.perf_counter() - t0
                        if collect_stats:
                            stats["score_calls"] += 1
                            stats["score_time_sec"] += dt
                        if scores.size == len(actions):
                            scores = np.where(np.isfinite(scores), scores, -1e9)
                            best_action = actions[int(np.argmax(scores))]
                    except Exception as exc:
                        # If scoring fails (e.g., feature/model mismatch), fall back to pure heuristic
                        # for the rest of the run to avoid mixing "topB seed" logic with heuristic scoring.
                        if not nsgbs_score_error_printed:
                            print(f"[NS-GBS] Scoring failed ({exc}); falling back to heuristic scoring.")
                            nsgbs_score_error_printed = True
                        use_nsgbs = False
                        nsgbs_scorer = None
                        actions = None
                        best_action = None

            if best_action is None:
                best_delta = -1e9
                for action in (actions if actions is not None else iter_actions()):
                    metric = score_action_heuristic(action)
                    if metric > best_delta:
                        best_delta = metric
                        best_action = action

            # Fallback: if no action found (e.g., all capped), assign highest remaining PRB to best UE
            if best_action is None:
                remaining = np.flatnonzero(winners < 0)
                if remaining.size == 0:
                    break
                z = int(remaining[0])
                cand = np.arange(N_UE) if max_prbs_per_ue is None else np.flatnonzero(k_assigned < int(max_prbs_per_ue))
                if mask_t is not None:
                    cand = cand[mask_t[cand]]
                if cand.size == 0:
                    break
                ue = int(cand[np.argmax(se_pred_k1[cand, z] / Rbar[cand])])
                best_action = ("fallback", ue, z)

            _, ue_sel, z_sel = best_action
            apply_assign(int(ue_sel), int(z_sel))
            assigned_cnt += 1

        # Optionally record per-PRB winners for this TTI
        if record_assignments and assignments_out is not None:
            try:
                assignments_out.append(np.array(winners, copy=True))
            except Exception:
                pass

        # Compute throughput or register HARQ TBs
        if snr_lin_time is not None:
            snr_true = np.asarray(snr_lin_time[t_idx], dtype=float)
        elif snr_lin is not None:
            snr_true = np.asarray(snr_lin, dtype=float)
        else:
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap)) - 1.0)
            snr_true = gamma

        if (harq_mgr is not None) and hasattr(harq_mgr, 'on_scheduled_blocks'):
            # Register TBs for HARQ manager; credit happens on feedback
            # Optionally apply DL power allocation (e.g., water-filling) before building SINR vectors
            winners_full = np.full(Z, -1, dtype=int)
            for ue in range(N_UE):
                if int(k_assigned[ue]) <= 0:
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                winners_full[li:ri+1] = ue

            snr_scaled = np.array(snr_true, copy=True)
            if str(dl_power_model).lower() == 'waterfill' and P_tot_dbm is not None:
                # Group-level (per-UE block) water-filling: same power per PRB within a UE block
                # This preserves single-MCS per block and reduces EESM penalty.
                blocks = []  # (ue, li, ri)
                for ue in range(N_UE):
                    if int(k_assigned[ue]) <= 0:
                        continue
                    li, ri = int(l_idx[ue]), int(r_idx[ue])
                    blocks.append((ue, li, ri))
                if blocks:
                    P_ref_dbm_eff = float(P_ref_dbm) if P_ref_dbm is not None else 0.0
                    P_ref_mW = 10.0 ** (P_ref_dbm_eff / 10.0)
                    P_tot_mW = 10.0 ** (float(P_tot_dbm) / 10.0)
                    pmin_mW = 0.0 if p_min_dbm is None else 10.0 ** (float(p_min_dbm) / 10.0)
                    pmax_mW = float('inf') if p_max_dbm is None else 10.0 ** (float(p_max_dbm) / 10.0)
                    # Build group gains (a_bar per PRB) and weights (k = PRBs in block)
                    a_g = []
                    k_g = []
                    for (ue, li, ri) in blocks:
                        k = (ri - li + 1)
                        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                        a_vec = np.maximum(1e-12, snr_true[ue, li:ri+1]) / P0
                        a_g.append(float(np.mean(a_vec)))
                        k_g.append(int(k))
                    a_g = np.asarray(a_g, dtype=float)
                    k_g = np.asarray(k_g, dtype=float)
                    # Handle infeasible lower bound: if sum k*pmin > P, relax pmin uniformly
                    sum_min = float(np.sum(k_g) * pmin_mW)
                    if pmin_mW > 0.0 and (sum_min > P_tot_mW):
                        pmin_mW = P_tot_mW / float(np.sum(k_g))
                    # Weighted water-filling on groups
                    def waterfill_groups(a_vec: np.ndarray, w_vec: np.ndarray, P: float, pmin: float, pmax: float) -> np.ndarray:
                        a_vec = np.asarray(a_vec, dtype=float)
                        w_vec = np.asarray(w_vec, dtype=float)
                        # Bisection on nu for sum w * p(nu) = P
                        lo, hi = 1e-12, max(1.0, a_vec.max() * 1e3)
                        def total(nu: float) -> float:
                            p = np.maximum(0.0, 1.0/nu - 1.0/np.maximum(a_vec, 1e-30))
                            if pmax < float('inf'):
                                p = np.minimum(p, pmax)
                            if pmin > 0.0:
                                p = np.maximum(p, pmin)
                            return float(np.sum(w_vec * p))
                        # If even at hi the total < P, reduce hi to meet
                        for _ in range(60):
                            mid = (lo + hi) * 0.5
                            s = total(mid)
                            if s > P:
                                lo = mid
                            else:
                                hi = mid
                        # Final p per group
                        nu = hi
                        p = np.maximum(0.0, 1.0/nu - 1.0/np.maximum(a_vec, 1e-30))
                        if pmax < float('inf'):
                            p = np.minimum(p, pmax)
                        if pmin > 0.0:
                            p = np.maximum(p, pmin)
                        # Normalize tiny residual due to clipping
                        s = float(np.sum(w_vec * p))
                        if s > 0 and abs(s - P) / P > 1e-3:
                            p *= (P / s)
                        return p
                    p_grp = waterfill_groups(a_g, k_g, P_tot_mW, pmin_mW, pmax_mW)
                    # Apply per-UE uniform PRB power
                    for idx, (ue, li, ri) in enumerate(blocks):
                        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                        scale = p_grp[idx] / P0
                        snr_scaled[ue, li:ri+1] = snr_true[ue, li:ri+1] * scale

            sched_info: Dict[int, Dict] = {}
            for ue in range(N_UE):
                k0 = int(k_assigned[ue])
                if k0 <= 0:
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                snr_vec_db = 10.0 * np.log10(np.maximum(snr_scaled[ue, li:ri + 1], 1e-12))
                sched_info[int(ue)] = {
                    'sinr_vec_db': snr_vec_db,
                    'n_prb': (ri - li + 1),
                    'eesm_beta_db': float(eesm_beta_db),
                    'li': li, 'ri': ri,
                }
            harq_mgr.on_scheduled_blocks(sched_info)
        else:
            # Legacy immediate throughput accumulation with optional DL power allocation
            # Build per-PRB base SNR for used PRBs
            winners_full = np.full(Z, -1, dtype=int)
            for ue in range(N_UE):
                if int(k_assigned[ue]) <= 0:
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                winners_full[li:ri+1] = ue

            snr_scaled = np.array(snr_true, copy=True)

            if str(dl_power_model).lower() == 'waterfill' and P_tot_dbm is not None:
                used = np.flatnonzero(winners_full >= 0)
                if used.size > 0:
                    P_ref_dbm_eff = float(P_ref_dbm) if P_ref_dbm is not None else 0.0
                    P_ref_mW = 10.0 ** (P_ref_dbm_eff / 10.0)
                    P_tot_mW = 10.0 ** (float(P_tot_dbm) / 10.0)
                    pmin_mW = 0.0 if p_min_dbm is None else 10.0 ** (float(p_min_dbm) / 10.0)
                    pmax_mW = float('inf') if p_max_dbm is None else 10.0 ** (float(p_max_dbm) / 10.0)

                    a = np.zeros(used.size, dtype=float)
                    for i, z in enumerate(used):
                        ue = int(winners_full[z])
                        base = max(1e-12, snr_true[ue, z])
                        # snr_true corresponds to P_ref_mW; if P_ref_mW==0, treat as 1 mW reference
                        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                        a[i] = base / P0

                    # Water-filling solver with box constraints
                    def waterfill(a_vec: np.ndarray, P: float, pmin: float, pmax: float) -> np.ndarray:
                        a_vec = np.asarray(a_vec, dtype=float)
                        n = a_vec.size
                        # Active set algorithm
                        active = np.ones(n, dtype=bool)
                        p = np.zeros(n, dtype=float)
                        # Initialize ignoring bounds
                        while True:
                            a_act = a_vec[active]
                            if a_act.size == 0:
                                break
                            inv_a = 1.0 / a_act
                            # Solve for nu: sum(max(0, 1/nu - 1/a)) = P_eff
                            # Using bisection on nu in (0, max(a))
                            lo, hi = 1e-12, max(1.0, a_act.max()*1e3)
                            for _ in range(40):
                                nu = (lo + hi) * 0.5
                                p_tmp = np.maximum(0.0, 1.0/nu - inv_a)
                                s = p_tmp.sum()
                                if s > P:
                                    lo = nu
                                else:
                                    hi = nu
                            p_act = np.maximum(0.0, 1.0/hi - inv_a)
                            # Apply box constraints
                            p_act = np.clip(p_act, pmin, pmax)
                            p[:] = 0.0
                            p[active] = p_act
                            # Check total power vs P with saturated elements removed
                            if abs(p.sum() - P) < 1e-6:
                                break
                            # If sum > P due to lower bounds, reduce active set of saturated lows
                            if p.sum() > P + 1e-6:
                                # Reduce those at pmin from active and re‑solve
                                mask = active.copy()
                                idxs = np.flatnonzero(active)
                                sat_low = (p[idxs] <= pmin + 1e-12)
                                if not np.any(sat_low):
                                    break
                                active[idxs[sat_low]] = False
                                P = max(0.0, P - np.sum(p[idxs[sat_low]]))
                                continue
                            # If sum < P due to upper bounds, remove highs and re‑distribute residual
                            resid = P - p.sum()
                            if resid <= 1e-6:
                                break
                            mask = active.copy()
                            idxs = np.flatnonzero(active)
                            sat_high = (p[idxs] >= pmax - 1e-12)
                            if not np.any(sat_high):
                                # Distribute tiny residual equally
                                p[idxs] += resid / float(len(idxs))
                                break
                            active[idxs[sat_high]] = False
                            P = resid
                        return p

                    p_used = waterfill(a, P_tot_mW, pmin_mW, pmax_mW)
                    # Scale SNRs
                    for i, z in enumerate(used):
                        ue = int(winners_full[z])
                        P0 = P_ref_mW if P_ref_mW > 0.0 else 1.0
                        scale = p_used[i] / P0
                        snr_scaled[ue, z] = snr_true[ue, z] * scale

            # Throughput accumulation per UE
            thr_i = np.zeros(N_UE, dtype=float)
            for ue in range(N_UE):
                k0 = int(k_assigned[ue])
                if k0 <= 0:
                    continue
                li, ri = int(l_idx[ue]), int(r_idx[ue])
                snr_vec = snr_scaled[ue, li:ri + 1]
                se_per_prb = _block_se_from_snr_vec(snr_vec, 1 if (str(dl_power_model).lower() in ('equal_prb','waterfill')) else (k0 if power_split else 1), use_mcs, mcs_params, eesm_beta_db)
                thr_i[ue] = (ri - li + 1) * se_per_prb * overhead_eff
            if record_ue_thr and ue_thr_out is not None:
                try:
                    ue_thr_out.append(np.array(thr_i, copy=True))
                except Exception:
                    pass
            sum_rate += thr_i.sum()
            Rbar = (1 - beta) * Rbar + beta * thr_i
            if harq_mgr is not None and hasattr(harq_mgr, 'on_scheduled'):
                scheduled = np.flatnonzero(k_assigned > 0)
                harq_mgr.on_scheduled(scheduled)

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    if collect_stats and stats_out is not None and stats is not None:
        # Derived NS-GBS timing stats (for complexity plots)
        try:
            score_calls = int(stats.get("score_calls", 0) or 0)
            actions_total = int(stats.get("actions_total", 0) or 0)
            score_time_sec = float(stats.get("score_time_sec", 0.0) or 0.0)
            t_total = int(T) if T else 0
            stats["avg_score_ms_per_call"] = (score_time_sec * 1000.0) / score_calls if score_calls > 0 else 0.0
            stats["avg_score_us_per_action"] = (score_time_sec * 1e6) / actions_total if actions_total > 0 else 0.0
            stats["score_ms_per_tti"] = (score_time_sec * 1000.0) / t_total if t_total > 0 else 0.0
            stats["actions_per_score_call"] = (float(actions_total) / float(score_calls)) if score_calls > 0 else 0.0
            stats["actions_per_tti"] = (float(actions_total) / float(t_total)) if t_total > 0 else 0.0
        except Exception:
            pass
        stats_out.clear()
        stats_out.update(stats)
    return avg_sum_rate_per_prb


def _group_ranges(Z: int, num_groups: int) -> list:
    """Split Z PRBs into num_groups contiguous groups; last group may be larger by at most 1 PRB."""
    num_groups = max(1, int(num_groups))
    base = Z // num_groups
    rem = Z % num_groups
    ranges = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def pf_schedule_baseline_subband(
    Z: int,
    T: int,
    beta: float,
    overhead_eff: float,
    use_mcs: bool,
    power_split: bool,
    mcs_params: Optional[Dict],
    num_groups: int,
    eesm_beta_db: float = 1.0,
    max_groups_per_ue: Optional[int] = None,
    use_marginal_delta: bool = True,
    # Inputs for scheduling metric and throughput
    snr_lin_prb: Optional[np.ndarray] = None,                      # [UE,Z]
    cap_prb: Optional[np.ndarray] = None,                          # [UE,Z] (Shannon)
    snr_lin_time_prb_metric: Optional[np.ndarray] = None,          # [T,UE,Z] metric (delayed) SNR
    snr_lin_time_prb_true: Optional[np.ndarray] = None,            # [T,UE,Z] true instantaneous SNR (for throughput)
    se_metric_time_subband: Optional[np.ndarray] = None,           # [T,UE,S] optional precomputed subband metric
) -> float:
    """
    Baseline with subband-level CQI: split Z PRBs into S groups, compute per-group SE metric
    (EESM+MCS if use_mcs else Shannon mean), then PF-allocate groups each TTI. Each assigned
    group uses a single MCS evaluated over the group's PRBs. If power_split is enabled, the
    UE's total PRB count across all assigned groups divides its SNR.
    Returns average sum SE per PRB (bits/s/Hz).
    """
    # Build groups
    groups = _group_ranges(Z, num_groups)
    S = len(groups)

    # Helper: compute subband metric from per-PRB SNR/SE
    def _metric_from_snr_mat(snr_mat: np.ndarray) -> np.ndarray:
        out = np.zeros((snr_mat.shape[0], S), dtype=float)
        for gi, (li, ri) in enumerate(groups):
            vec = snr_mat[:, li:ri + 1]
            # Per-group per-PRB SE at k=1
            if use_mcs:
                # EESM then MCS mapping
                sinr_db_vec = 10.0 * np.log10(np.maximum(vec, 1e-12))
                sinr_eff_db = effective_sinr_eesm(sinr_db_vec, beta_db=float(eesm_beta_db), axis=-1)
                out[:, gi] = sinr_to_se_mcs(sinr_eff_db, table=mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy")
            else:
                out[:, gi] = np.log2(1.0 + np.maximum(vec, 0.0)).mean(axis=1)
        return out

    # Prepare static subband metric if no time series given
    if se_metric_time_subband is None:
        if snr_lin_prb is not None:
            se_metric_sub = _metric_from_snr_mat(np.asarray(snr_lin_prb, dtype=float))  # [UE,S]
        elif cap_prb is not None and (not use_mcs):
            # Average Shannon per group
            se_metric_sub = np.zeros((cap_prb.shape[0], S), dtype=float)
            for gi, (li, ri) in enumerate(groups):
                se_metric_sub[:, gi] = np.asarray(cap_prb[:, li:ri + 1]).mean(axis=1)
        else:
            # Fallback: invert Shannon from cap to get SNR
            if cap_prb is None:
                raise ValueError("pf_schedule_baseline_subband requires snr_lin_prb or cap_prb")
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_prb)) - 1.0)
            se_metric_sub = _metric_from_snr_mat(gamma)

    # Resolve N_UE
    if se_metric_time_subband is not None:
        N_UE = se_metric_time_subband.shape[1]
    elif snr_lin_prb is not None:
        N_UE = int(np.asarray(snr_lin_prb).shape[0])
    elif cap_prb is not None:
        N_UE = int(np.asarray(cap_prb).shape[0])
    else:
        # Time-varying but metric is to be built from snr_lin_time_prb_metric per TTI
        if snr_lin_time_prb_metric is None:
            raise ValueError("Need snr_lin_time_prb_metric or se_metric_time_subband or static snr/cap")
        N_UE = int(np.asarray(snr_lin_time_prb_metric).shape[1])
    Rbar = np.full(N_UE, 1e-3)
    sum_rate = 0.0

    for t_idx in range(T):
        # Scheduling metric for this TTI
        if se_metric_time_subband is not None:
            metric_se = se_metric_time_subband[t_idx]  # [UE,S]
            snr_metric_mat = None
        elif snr_lin_time_prb_metric is not None:
            # Build subband metric from delayed SNR for this TTI
            snr_metric_mat = np.asarray(snr_lin_time_prb_metric[t_idx], dtype=float)
            metric_se = _metric_from_snr_mat(snr_metric_mat)
        else:
            metric_se = se_metric_sub
            snr_metric_mat = None

        # Allocate groups with PF using either per-group metric or marginal ΔSE
        winners = np.full(S, -1, dtype=int)
        if not use_marginal_delta:
            metric = metric_se / Rbar.reshape(-1, 1)
            if max_groups_per_ue is None:
                winners = np.argmax(metric, axis=0)
            else:
                counts = np.zeros(N_UE, dtype=int)
                best_vals = metric.max(axis=0)
                order_g = np.argsort(-best_vals)
                for idx in order_g:
                    ue_best = int(np.argmax(metric[:, idx]))
                    if counts[ue_best] < int(max_groups_per_ue):
                        winners[idx] = ue_best
                        counts[ue_best] += 1
                    else:
                        sorted_ues = np.argsort(-metric[:, idx])
                        chosen = -1
                        for u in sorted_ues:
                            if counts[u] < int(max_groups_per_ue):
                                chosen = int(u)
                                break
                        if chosen < 0:
                            chosen = int(np.argmax(metric[:, idx]))
                        winners[idx] = chosen
                        counts[chosen] += 1
        else:
            # Marginal ΔSE allocation using predicted SNR mat if available
            assigned_groups = [[] for _ in range(N_UE)]
            k_prb_assigned = np.zeros(N_UE, dtype=int)
            block_se_pred = np.zeros(N_UE, dtype=float)
            counts = np.zeros(N_UE, dtype=int)
            remaining = set(range(S))
            while remaining:
                best_delta = -1e9
                best_pair = None  # (ue, gi)
                # Try each remaining group, pick best PF metric (Δ/Rbar)
                for gi in list(remaining):
                    li, ri = groups[gi]
                    size_g = ri - li + 1
                    for ue in range(N_UE):
                        if (max_groups_per_ue is not None) and (counts[ue] >= int(max_groups_per_ue)):
                            continue
                        k0 = int(k_prb_assigned[ue])
                        # Predicted SNR vector for UE on union of assigned PRBs (+candidate group)
                        if snr_metric_mat is not None:
                            if k0 > 0:
                                # Build current mask
                                mask = np.zeros(Z, dtype=bool)
                                for gprev in assigned_groups[ue]:
                                    l0, r0 = groups[gprev]
                                    mask[l0:r0+1] = True
                                snr_vec_old = snr_metric_mat[ue, mask]
                            else:
                                snr_vec_old = None
                            # New vector includes candidate group
                            mask_new = np.zeros(Z, dtype=bool)
                            if k0 > 0:
                                for gprev in assigned_groups[ue]:
                                    l0, r0 = groups[gprev]
                                    mask_new[l0:r0+1] = True
                            mask_new[li:ri+1] = True
                            snr_vec_new = snr_metric_mat[ue, mask_new]
                            se_old = _block_se_from_snr_vec(snr_vec_old, k0 if (k0>0 and power_split) else 1, use_mcs, mcs_params, eesm_beta_db) if (k0>0) else 0.0
                            k_new = k0 + size_g
                            se_new = _block_se_from_snr_vec(snr_vec_new, k_new if power_split else 1, use_mcs, mcs_params, eesm_beta_db)
                        else:
                            # Fallback: use per-group metric only
                            se_old = float(block_se_pred[ue]) if k0>0 else 0.0
                            k_new = k0 + size_g
                            # Approximate new per-PRB SE as average of existing block_se_pred and this group's per-PRB metric
                            se_g = float(metric_se[ue, gi])
                            se_new = (k0 * se_old + size_g * se_g) / float(k_new)
                            if power_split and use_mcs:
                                # Rough penalty for power split in absence of SNR
                                pass
                        delta = k_new * se_new - k0 * se_old
                        metric_pf = delta / Rbar[ue]
                        if metric_pf > best_delta:
                            best_delta = metric_pf
                            best_pair = (ue, gi, k_new, se_new)
                if best_pair is None:
                    break
                ue_sel, gi_sel, k_new_sel, se_new_sel = best_pair
                winners[gi_sel] = ue_sel
                remaining.remove(gi_sel)
                counts[ue_sel] += 1
                # Update assigned sets
                assigned_groups[ue_sel].append(gi_sel)
                k_prb_assigned[ue_sel] = k_new_sel
                block_se_pred[ue_sel] = se_new_sel

        # Throughput accumulation
        thr_i = np.zeros(N_UE, dtype=float)
        # Count total PRBs per UE for power split
        if power_split:
            prbs_per_ue = np.zeros(N_UE, dtype=int)
            for gi, (li, ri) in enumerate(groups):
                prbs_per_ue[winners[gi]] += (ri - li + 1)
        else:
            prbs_per_ue = np.ones(N_UE, dtype=int)

        # Select SNR field for this TTI (true for throughput)
        if snr_lin_time_prb_true is not None:
            snr_true = np.asarray(snr_lin_time_prb_true[t_idx], dtype=float)  # [UE,Z]
        elif snr_lin_prb is not None:
            snr_true = np.asarray(snr_lin_prb, dtype=float)
        else:
            # Invert Shannon
            gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_prb)) - 1.0)
            snr_true = gamma

        for gi, (li, ri) in enumerate(groups):
            ue = int(winners[gi])
            k_prb = int(prbs_per_ue[ue]) if power_split else 1
            snr_vec = snr_true[ue, li:ri + 1]
            se_per_prb = _block_se_from_snr_vec(snr_vec, k_prb, use_mcs, mcs_params, eesm_beta_db)
            thr_i[ue] += (ri - li + 1) * se_per_prb * overhead_eff

        sum_rate += thr_i.sum()
        Rbar = (1 - beta) * Rbar + beta * thr_i

    avg_sum_rate_per_prb = sum_rate / (T * Z)
    return avg_sum_rate_per_prb

# -----------------------
# Experiment harness
# -----------------------
def run_once(config: Dict) -> Dict:
    """Single experiment orchestration with lower cyclomatic complexity."""
    rng = np.random.default_rng(config["seed"])
    N_UE, T = config["N_UE"], config["T"]

    # Radio map and UEs
    R_xyz_dbm, X, Y, Z = select_radio_map(config)
    ue_pos = generate_ue_positions(N_UE, X, Y, rng)

    # Geometry/beam, noise/bandwidth, power settings
    L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue = compute_geometry_and_beam(config, X, Y, ue_pos)
    noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)
    P_tx_per_ue_dbm = apply_open_loop_power_control(config, L_fs_per_ue, G_rx_per_ue)

    # Static snapshot (also used as reference when no time-variation)
    cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
        R_xyz_dbm, ue_pos,
        P_tx_dbm=P_tx_per_ue_dbm,
        L_fs_db=L_fs_per_ue,
        G_rx_db=G_rx_per_ue,
        shadow_db_std=config["shadow_std_db"],
        N0_dbm=noise_dbm,
        rx_nf_db=config.get("rx_nf_db", 0.0),
        impl_loss_db=config.get("impl_loss_db", 0.0),
        seed=config["seed"],
        elevation_deg=elev_deg_per_ue,
        channel_model=config.get("channel_model", "3gpp_ntn"),
        channel_params=config.get("channel_params"),
        channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
    )

    # Estimation error: optional static predicted metric for RadioMap scheduler
    metric_override = compute_metric_override_static_if_needed(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, noise_dbm, elev_deg_per_ue
    )

    # Orbit dynamics (measurement geometry) if enabled; no HO gating in minimal DL
    ue_mask_time = None
    events = None
    if config.get("enable_orbit_dynamics", False):
        orbit_model_meas = OrbitModel(config, X, Y)
    else:
        orbit_model_meas = None

    # One-time orbit mode prompt for clarity
    try:
        if orbit_model_meas is None:
            print("[Orbit] Dynamics disabled: using static geometry (no time-varying orbit).")
        else:
            tle_name = str(config.get("tle_name", "SAT"))
            start = str(config.get("orbit_start_datetime", "t0"))
            auto_ref = bool(config.get("auto_ref_from_tle", False))
            ref_lat = config.get("ref_lat_deg", None)
            ref_lon = config.get("ref_lon_deg", None)
            ref_txt = f", ref=({ref_lat:.4f}, {ref_lon:.4f})" if (isinstance(ref_lat, (int, float)) and isinstance(ref_lon, (int, float))) else ""
            print(f"[Orbit] Using Skyfield/TLE orbit: {tle_name} (start={start}), auto_ref={auto_ref}{ref_txt}.")
    except Exception:
        pass

    # Optional time variation
    time_series = build_time_variation_if_enabled(
        config, R_xyz_dbm, ue_pos, L_fs_per_ue, G_rx_per_ue, elev_deg_per_ue, P_tx_per_ue_dbm, noise_dbm, rng,
        None if config.get("enable_time_varying", False) else metric_override,
        orbit_model=orbit_model_meas,
    )
    # Register 3GPP MCS tables from file if provided
    try:
        if config.get("mcs_3gpp_table_path"):
            register_mcs_tables_from_file(config.get("mcs_3gpp_table_path"))
    except Exception as e:
        print(f"[WARN] Failed to load 3GPP MCS tables: {e}")
    # Register BLER curves if provided
    try:
        if config.get("bler_curve_path"):
            register_bler_curves_from_file(config.get("bler_curve_path"))
    except Exception as e:
        print(f"[WARN] Failed to load BLER curves: {e}")
    mcs_params = {
        "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
        "mcs_table": config.get("csi_mcs_table", "legacy"),
        "residual_freq_hz": config.get("residual_freq_hz", 0.0),
        "scs_khz": config.get("scs_khz", 30),
    }

    if time_series is not None:
        # Apply CSI delay to scheduler metrics (not to actual SNR)
        def delay_series(arr: np.ndarray, d: int) -> np.ndarray:
            if d <= 0:
                return arr
            T0 = arr.shape[0]
            out = np.empty_like(arr)
            for t in range(T0):
                src = max(0, t - d)
                out[t] = arr[src]
            return out
        def hold_series(arr: np.ndarray, period: int, offset: int = 0) -> np.ndarray:
            """Hold-last across time axis 0 given a reporting period/offset."""
            if period is None or period <= 1:
                return arr
            T0 = arr.shape[0]
            out = np.empty_like(arr)
            last = None
            for t in range(T0):
                if ((t - offset) % period) == 0:
                    out[t] = arr[t]
                    last = arr[t]
                else:
                    out[t] = arr[t] if last is None else last
            return out

        baseline_delay = int(config.get("baseline_csi_delay_ttis", 0))
        rm_delay = int(config.get("rm_csi_delay_ttis", 0))

        # Wideband metric (baseline)
        se_time_wb = delay_series(time_series["se_time_wb"], baseline_delay)

        # RadioMap metric (per PRB)
        se_time_rm = delay_series(time_series["se_time_rm"], rm_delay)

        # Baseline per-PRB SE metric derived from instantaneous SNR, then delay and optional CQI quant
        se_time_base = np.empty_like(time_series["se_time_rm"])  # [T, UE, Z]
        for tt in range(time_series["snr_time"].shape[0]):
            # Baseline fixed: use CQI quantization for metric
            se_time_base[tt] = snr_to_se_sched(
                time_series["snr_time"][tt], config.get("use_mcs", False), mcs_params,
                enable_cqi_quant=True,
                cqi_table=config.get("csi_mcs_table", "nr_256qam")
            )
        se_time_base = delay_series(se_time_base, baseline_delay)
        # Optional CQI reporting periodicity (hold-last), decoupled per-path
        period = int(config.get("cqi_period_ttis", 0) or 0)
        offset = int(config.get("cqi_offset_ttis", 0) or 0)
        if period and period > 1:
            if bool(config.get("enable_cqi_periodicity_base", False)):
                se_time_wb = hold_series(se_time_wb, period, offset)
                se_time_base = hold_series(se_time_base, period, offset)
            if bool(config.get("enable_cqi_periodicity_rm", False)):
                se_time_rm = hold_series(se_time_rm, period, offset)
        # Keep tau/fd for downstream users
        tau_time = time_series.get("tau_time")
        fd_time = time_series.get("fd_time")
        # No HO/RACH gating events in minimal DL

        # Optional HARQ (Stage-2 deferral or full Stage-3-like)
        harq_stats_base = None
        harq_stats_map = None
        harq_mgr_base = None
        harq_mgr_map = None
        if bool(config.get("enable_harq_full", False)):
            harq_mgr_base = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
            harq_mgr_map = HarqManagerFull(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                config=config,
            )
        elif bool(config.get("enable_harq_deferral", False)):
            harq_mgr_base = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )
            harq_mgr_map = HarqManager(
                num_ue=N_UE,
                num_procs=int(config.get("harq_max_procs", 16)),
                ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
            )
        # Baseline: contiguous-block PF using per-PRB metric
        _rec_base = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all"))
        _rec_base_thr = bool(config.get("record_ue_thr", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("base", "both", "all"))
        assignments_base = [] if _rec_base else None
        ue_thr_base = [] if _rec_base_thr else None
        # Fairness: keep the 3GPP-like baseline independent of NS-GBS/MLP/ISAB mode.
        _cfg_base = dict(config)
        _cfg_base["scheduler_kind"] = "heuristic"
        _cfg_base["nsgbs_model_path"] = None
        # Avoid polluting NS-GBS stats/datasets when computing the baseline.
        _cfg_base["nsgbs_collect_stats"] = False
        _cfg_base["nsgbs_stats_out"] = None
        _cfg_base["nsgbs_collect_dataset"] = False
        _cfg_base["nsgbs_dataset_out"] = None
        base_se_default = pf_schedule_radiomap_blocks(
                cap, T, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
                mcs_params=mcs_params,
                se_metric_time=se_time_base,
                snr_lin_time=time_series["snr_time"],
                eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=None,
                harq_mgr=harq_mgr_base,
                dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
                P_tot_dbm=config.get("baseline_P_tot_dbm", None),
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=config.get("baseline_p_min_dbm", None),
                p_max_dbm=config.get("baseline_p_max_dbm", None),
                record_assignments=_rec_base,
                assignments_out=assignments_base,
                record_ue_thr=_rec_base_thr,
                ue_thr_out=ue_thr_base,
                config=_cfg_base,
            )
        

        base_se_subband = None
        # RadioMap: contiguous-block PF with per-PRB metric
        if True:
            _rec_rm = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
            _rec_rm_thr = bool(config.get("record_ue_thr", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
            assignments_rm = [] if _rec_rm else None
            ue_thr_rm = [] if _rec_rm_thr else None
            map_se = pf_schedule_radiomap_blocks(
                cap, T, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None if metric_override is None else metric_override,
                max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
                mcs_params=mcs_params,
                se_metric_time=se_time_rm,
                snr_lin_time=time_series["snr_time"],
                eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=None,
                harq_mgr=harq_mgr_map,
                dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
                P_tot_dbm=config.get("rm_P_tot_dbm", None),
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=config.get("rm_p_min_dbm", None),
                p_max_dbm=config.get("rm_p_max_dbm", None),
                record_assignments=_rec_rm,
                assignments_out=assignments_rm,
                record_ue_thr=_rec_rm_thr,
                ue_thr_out=ue_thr_rm,
                config=config,
            )
            sched_stats = None
        else:
            pass
        # Collect HARQ statistics if available
        if harq_mgr_base is not None and hasattr(harq_mgr_base, 'get_stats'):
            try:
                harq_stats_base = harq_mgr_base.get_stats()
            except Exception:
                harq_stats_base = None
        if harq_mgr_map is not None and hasattr(harq_mgr_map, 'get_stats'):
            try:
                harq_stats_map = harq_mgr_map.get_stats()
            except Exception:
                harq_stats_map = None
    else:
        # Baseline: contiguous-block PF using per-PRB metric
        # Fairness: keep the 3GPP-like baseline independent of NS-GBS/MLP/ISAB mode.
        _cfg_base = dict(config)
        _cfg_base["scheduler_kind"] = "heuristic"
        _cfg_base["nsgbs_model_path"] = None
        _cfg_base["nsgbs_collect_stats"] = False
        _cfg_base["nsgbs_stats_out"] = None
        _cfg_base["nsgbs_collect_dataset"] = False
        _cfg_base["nsgbs_dataset_out"] = None
        base_se_default = pf_schedule_radiomap_blocks(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=None,
            max_prbs_per_ue=config.get("baseline_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=rng,
            dl_power_model=str(config.get("baseline_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("baseline_P_tot_dbm", None),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("baseline_p_min_dbm", None),
            p_max_dbm=config.get("baseline_p_max_dbm", None),
            config=_cfg_base,
        )
        
        base_se_subband = None
        # RadioMap: contiguous-block PF with per-PRB metric
        _rec_rm2 = bool(config.get("record_assignments", False)) and (str(config.get("record_assignments_target", "rm")).lower() in ("rm", "both", "all"))
        assignments_rm2 = [] if _rec_rm2 else None
        map_se = pf_schedule_radiomap_blocks(
            cap, T, beta=config["pf_beta"],
            snr_lin=snr_lin,
            overhead_eff=config.get("overhead_eff", 1.0),
            use_mcs=config.get("use_mcs", False),
            power_split=config.get("power_split", False),
            se_metric_override=metric_override,
            max_prbs_per_ue=config.get("rm_max_prbs_per_ue", config.get("max_prbs_per_ue")),
            mcs_params=mcs_params,
            se_metric_time=None,
            snr_lin_time=None,
            eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
            require_contiguous=bool(config.get("sched_require_contiguous", True)),
            rng=rng,
            ue_mask_time=None,
            dl_power_model=str(config.get("rm_dl_power_model", "equal_prb")),
            P_tot_dbm=config.get("rm_P_tot_dbm", None),
            P_ref_dbm=config.get("P_tx_dbm"),
            p_min_dbm=config.get("rm_p_min_dbm", None),
            p_max_dbm=config.get("rm_p_max_dbm", None),
            record_assignments=_rec_rm2,
            assignments_out=assignments_rm2,
            config=config,
        )
        sched_stats = None
        harq_stats_base = None
        harq_stats_map = None

    # Optional JSON report with per-UE throughput/fairness and events
    # Compute system bandwidth for throughput reporting
    try:
        sys_bw_hz = float(prb_bw_hz) * float(cap.shape[1])
    except Exception:
        sys_bw_hz = float(config.get("scs_khz", 30.0)) * 1e3 * 12.0 * float(cap.shape[1])

    report = {
        "avg_se_baseline_default": base_se_default,
        "avg_se_radiomap": map_se,
        "improvement_vs_default_pct": (map_se - base_se_default) / max(1e-9, base_se_default) * 100.0,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "cap": cap,
        "cap_wb": cap_wb,
        "snr_lin": snr_lin,
        "snr_lin_wb": snr_lin_wb,
        # Bandwidth/throughput metrics
        "prb_bw_hz": float(prb_bw_hz),
        "system_bandwidth_hz": float(sys_bw_hz),
        "total_throughput_baseline_bps": float(base_se_default * sys_bw_hz),
        "total_throughput_radiomap_bps": float(map_se * sys_bw_hz),
        # Optional dynamics for downstream consumers
        "tau_time": None if time_series is None else time_series.get("tau_time"),
        "fd_time": None if time_series is None else time_series.get("fd_time"),
        "sched_stats": None,
        "harq_stats_base": harq_stats_base,
        "harq_stats_map": harq_stats_map,
        "nsgbs_stats": config.get("nsgbs_stats_out"),
    }

    # Attach PRB assignment timeline and per-UE throughput if recorded
    try:
        if 'assignments_rm' in locals() and assignments_rm is not None and len(assignments_rm) > 0:
            report["assignments_rm"] = np.stack(assignments_rm, axis=0)
        if 'assignments_base' in locals() and assignments_base is not None and len(assignments_base) > 0:
            report["assignments_base"] = np.stack(assignments_base, axis=0)
        if 'ue_thr_rm' in locals() and ue_thr_rm is not None and len(ue_thr_rm) > 0:
            thr_mat = np.stack(ue_thr_rm, axis=0)  # [T,UE]
            report["ue_thr_time_rm"] = thr_mat
            # Convert to average per-UE SE per PRB: sum over time / (T*Z)
            report["per_ue_se_rm_avg"] = (np.sum(thr_mat, axis=0) / float(max(1, config.get("T", T)) * cap.shape[1])).tolist()
        if 'ue_thr_base' in locals() and ue_thr_base is not None and len(ue_thr_base) > 0:
            thr_mat_b = np.stack(ue_thr_base, axis=0)
            report["ue_thr_time_base"] = thr_mat_b
            report["per_ue_se_base_avg"] = (np.sum(thr_mat_b, axis=0) / float(max(1, config.get("T", T)) * cap.shape[1])).tolist()
    except Exception:
        pass

    # Attach PRB assignment timeline for RM if recorded
    try:
        if 'assignments_rm' in locals() and assignments_rm is not None and len(assignments_rm) > 0:
            report["assignments_rm"] = np.stack(assignments_rm, axis=0)
        elif 'assignments_rm2' in locals() and assignments_rm2 is not None and len(assignments_rm2) > 0:
            report["assignments_rm"] = np.stack(assignments_rm2, axis=0)
    except Exception:
        pass

    # Compute per-UE avg SE (goodput) from acked bits if available
    try:
        re_per_prb = re_per_prb_from_config(config)
        T_total = int(config.get("T", T))
        Z_total = cap.shape[1]
        def per_ue_avg_se(hs):
            if not hs or not isinstance(hs, dict) or 'acked_bits_per_ue' not in hs:
                return None
            bits = np.asarray(hs['acked_bits_per_ue'], dtype=float)
            return (bits / float(max(1, re_per_prb) * T_total * Z_total)).tolist()
        se_ue_base = per_ue_avg_se(harq_stats_base)
        se_ue_map = per_ue_avg_se(harq_stats_map)
        report["per_ue_avg_se_base"] = se_ue_base
        report["per_ue_avg_se_map"] = se_ue_map
        # Map per-UE SE to throughput (bps) using system bandwidth
        if se_ue_base is not None:
            report["per_ue_throughput_baseline_bps"] = (np.asarray(se_ue_base, dtype=float) * sys_bw_hz).tolist()
            report["avg_ue_throughput_baseline_bps"] = float(np.mean(report["per_ue_throughput_baseline_bps"]))
        else:
            # Fallback: equal-share approximation
            N_UE_eff = max(1, int(config.get("N_UE", cap.shape[0])))
            report["avg_ue_throughput_baseline_bps"] = float((base_se_default * sys_bw_hz) / N_UE_eff)
            # Try recorded per-UE SE if available
            try:
                if 'per_ue_se_base_avg' in report:
                    p = np.asarray(report['per_ue_se_base_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_baseline_bps"] = p.tolist()
                    report["avg_ue_throughput_baseline_bps"] = float(np.mean(p))
            except Exception:
                pass
        if se_ue_map is not None:
            report["per_ue_throughput_radiomap_bps"] = (np.asarray(se_ue_map, dtype=float) * sys_bw_hz).tolist()
            report["avg_ue_throughput_radiomap_bps"] = float(np.mean(report["per_ue_throughput_radiomap_bps"]))
        else:
            N_UE_eff = max(1, int(config.get("N_UE", cap.shape[0])))
            report["avg_ue_throughput_radiomap_bps"] = float((map_se * sys_bw_hz) / N_UE_eff)
            try:
                if 'per_ue_se_rm_avg' in report:
                    p = np.asarray(report['per_ue_se_rm_avg'], dtype=float) * sys_bw_hz
                    report["per_ue_throughput_radiomap_bps"] = p.tolist()
                    report["avg_ue_throughput_radiomap_bps"] = float(np.mean(p))
            except Exception:
                pass
        # Jain's fairness index
        def jain(x):
            if not x:
                return None
            arr = np.asarray(x, dtype=float)
            s = np.sum(arr)
            s2 = np.sum(arr * arr)
            n = arr.size
            return float((s * s) / max(1e-12, n * s2)) if s2 > 0 else 0.0
        report["fairness_jain_base"] = jain(se_ue_base) if se_ue_base is not None else None
        report["fairness_jain_map"] = jain(se_ue_map) if se_ue_map is not None else None
    except Exception:
        pass

    # Optionally write JSON to output directory
    try:
        if bool(config.get("write_json_report", False)):
            out_dir = config.get("plot_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            name = config.get("report_basename", "summary")
            path = os.path.join(out_dir, f"{name}.json")
            def serialize(obj):
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                raise TypeError
            with open(path, 'w') as f:
                json.dump(report, f, default=serialize)
    except Exception as e:
        print(f"[WARN] Failed to write JSON report: {e}")

    return report

def run_many(config: Dict, seeds: np.ndarray) -> Dict:
    base_def_list, map_list, imp_def_list = [], [], []
    for s in seeds:
        c2 = dict(config)
        c2["seed"] = int(s)
        out = run_once(c2)
        base_def_list.append(out["avg_se_baseline_default"])
        map_list.append(out["avg_se_radiomap"])
        imp_def_list.append(out["improvement_vs_default_pct"])
    return {
        "baseline_default": np.array(base_def_list),
        "radiomap": np.array(map_list),
        "improvement_vs_default_pct": np.array(imp_def_list),
    }

# -----------------------
# Constellation (multi-satellite) runner
# -----------------------
def run_constellation(config: Dict) -> Dict:
    """Multi-satellite coverage with independent per-satellite scheduling.

    - Reads a TLE catalog (docs/DTC_tle.txt) and builds a Skyfield constellation.
    - At each TTI: compute geometry per candidate sat; associate UEs (with optional HO);
      run per-satellite PF (RadioMap blocks + wideband baseline) on the served UE subset.
    - Inter-satellite interference: not modeled.
    """
    rng = np.random.default_rng(config["seed"])
    N_UE, T = int(config["N_UE"]), int(config["T"]) 

    # Radio map and UEs
    R_xyz_dbm, X, Y, Z = select_radio_map(config)
    ue_pos = generate_ue_positions(N_UE, X, Y, rng)

    # Noise and power
    noise_dbm, prb_bw_hz = resolve_noise_and_prb_bw(config)
    P_tx_dbm = apply_open_loop_power_control(config, 0.0, 0.0)  # returns config["P_tx_dbm"]

    # Register 3GPP MCS tables and optional BLER curves for HARQ/MCS selection
    try:
        if config.get("mcs_3gpp_table_path"):
            register_mcs_tables_from_file(config.get("mcs_3gpp_table_path"))
    except Exception as e:
        print(f"[WARN] Failed to load 3GPP MCS tables: {e}")
    try:
        if config.get("bler_curve_path"):
            register_bler_curves_from_file(config.get("bler_curve_path"))
    except Exception as e:
        print(f"[WARN] Failed to load BLER curves: {e}")

    # Constellation orbit
    orbit = ConstellationOrbit(config, X, Y)

    # HO/association state
    assoc_metric_kind = str(config.get("association_metric", "snr_wb")).lower()
    ho_enabled = bool(config.get("ho_enabled", True))
    ho_hyst_db = float(config.get("ho_hyst_db", 2.0))
    ho_ttt = int(config.get("ho_ttt_ttis", 20))
    min_elev = float(config.get("min_elev_deg", 5.0))
    serving = np.full(N_UE, -1, dtype=int)
    ho_timer = np.zeros(N_UE, dtype=int)
    # Current metric in dB scale for HO comparison
    curr_metric_db = np.full(N_UE, -1e9, dtype=float)
    # HO/outage logs
    ho_events: list = [[] for _ in range(N_UE)]
    outage_ttis = np.zeros(N_UE, dtype=int)
    include_trace = bool(config.get("include_serving_trace", False))
    serving_trace = [] if include_trace else None

    # Time-varying Radio Map (optional)
    R_base_t = R_xyz_dbm.copy()
    R_t = R_base_t
    vx, vy = config.get("rm_drift_px", (0, 0))
    flicker = float(config.get("rm_flicker_db_std", 0.0))
    flicker_kind = str(config.get("rm_flicker_kind", "global")).lower().strip()
    flicker_dist = str(config.get("rm_flicker_dist", "rectified")).lower().strip()
    enable_tv = bool(config.get("enable_time_varying", False))

    # KPI accumulators
    sum_rate_rm = 0.0
    sum_rate_base_def = 0.0
    kpi_per_sat: Dict[int, Dict] = {}
    # Per-satellite HARQ managers (persist across TTIs)
    harq_base_by_sat: Dict[int, HarqManagerFull] = {}
    harq_rm_by_sat: Dict[int, HarqManagerFull] = {}

    # For optional per-UE throughput (debug): not recording HARQ here
    # Progress bar for constellation TTI loop
    show_progress = bool(config.get("show_progress", True))
    leave_bar = bool(config.get("progress_leave", True))
    pbar_iter = tqdm(range(T), desc="Constellation", unit="TTI", disable=not show_progress,
                     leave=leave_bar,
                     bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for t_idx in pbar_iter:
        if enable_tv:
            if t_idx > 0 and (vx or vy):
                R_base_t = np.roll(R_base_t, shift=(int(vx), int(vy), 0), axis=(0, 1, 2))
            if flicker > 0.0:
                if flicker_dist in ("gaussian", "normal", "signed"):
                    def sample_delta(shape):
                        return rng.normal(0.0, flicker, size=shape)
                elif flicker_dist in ("rectified", "relu", "positive"):
                    def sample_delta(shape):
                        return np.maximum(0.0, rng.normal(0.0, flicker, size=shape))
                elif flicker_dist in ("abs", "absolute", "half_normal"):
                    def sample_delta(shape):
                        return np.abs(rng.normal(0.0, flicker, size=shape))
                else:
                    raise ValueError(f"Unknown rm_flicker_dist '{flicker_dist}'.")
                if flicker_kind in ("global", "scalar"):
                    R_t = R_base_t + float(sample_delta(None))
                elif flicker_kind in ("pixel", "per_pixel", "xy", "wideband_xy"):
                    delta = sample_delta((R_base_t.shape[0], R_base_t.shape[1], 1))
                    R_t = R_base_t + delta
                elif flicker_kind in ("prb", "per_prb", "z"):
                    delta = sample_delta((1, 1, R_base_t.shape[2]))
                    R_t = R_base_t + delta
                elif flicker_kind in ("element", "per_element", "xyz"):
                    R_t = R_base_t + sample_delta(R_base_t.shape)
                else:
                    raise ValueError(f"Unknown rm_flicker_kind '{flicker_kind}'.")
            else:
                R_t = R_base_t

        cand = orbit.candidate_indices_at(t_idx)
        if not cand:
            continue

        # Per-sat caches
        caps: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # (cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb), but we only keep some
        snr_lin_s: Dict[int, np.ndarray] = {}
        snr_wb_s: Dict[int, np.ndarray] = {}
        cap_s: Dict[int, np.ndarray] = {}
        prx_dbm_s: Dict[int, np.ndarray] = {}
        elev_s: Dict[int, np.ndarray] = {}

        # Build geometry and capacities per candidate sat
        # Unified PRB cap and power model for fair baseline vs RM in constellation mode
        prb_cap_unified = int(config.get("constellation_prb_cap", config.get("rm_max_prbs_per_ue", 20)))
        dlpm = str(config.get("rm_dl_power_model", "equal_prb"))
        Ptot = config.get("rm_P_tot_dbm", None)
        pmin = config.get("rm_p_min_dbm", None)
        pmax = config.get("rm_p_max_dbm", None)

        for si in cand:
            L_fs, G_rx, tau_s_arr, fd_hz_arr, elev = orbit.geometry_for_sat(ue_pos, si, t_idx)
            # Cap/SNR (per UE, PRB). Channel per sat per t; no inter-sat interference.
            cap, cap_wb, P_rx_dbm, I_total_dbm, snr_lin, snr_lin_wb = compute_caps(
                R_t, ue_pos,
                P_tx_dbm=P_tx_dbm,
                L_fs_db=L_fs,
                G_rx_db=G_rx,
                shadow_db_std=config["shadow_std_db"],
                N0_dbm=noise_dbm,
                rx_nf_db=config.get("rx_nf_db", 0.0),
                impl_loss_db=config.get("impl_loss_db", 0.0),
                seed=config["seed"],
                elevation_deg=elev,
                channel_model=config.get("channel_model", "3gpp_ntn"),
                channel_params=config.get("channel_params"),
                channel_profile=config.get("ntn_channel_profile", "s_band_handheld_urban"),
            )
            snr_lin_s[si] = snr_lin
            snr_wb_s[si] = snr_lin_wb
            cap_s[si] = cap
            prx_dbm_s[si] = P_rx_dbm
            elev_s[si] = elev

        # Association + (optional) HO
        # Compute best sat per UE based on metric and min elevation
        best_sat = np.full(N_UE, -1, dtype=int)
        best_metric_db = np.full(N_UE, -1e9, dtype=float)
        for si in cand:
            elev = elev_s[si]
            vis_mask = elev >= min_elev
            if assoc_metric_kind == 'snr_wb':
                met = snr_wb_s[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))
            elif assoc_metric_kind == 'prx_dbm':
                met_db = prx_dbm_s[si]
            else:
                # default to snr_wb
                met = snr_wb_s[si]
                met_db = 10.0 * np.log10(np.maximum(1e-12, met))
            # Apply visibility mask
            met_db = np.where(vis_mask, met_db, -1e9)
            take = met_db > best_metric_db
            best_metric_db = np.where(take, met_db, best_metric_db)
            best_sat = np.where(take, si, best_sat)

        # Update serving with HO policy
        for ue in range(N_UE):
            s_old = int(serving[ue])
            met_old = float(curr_metric_db[ue])
            b = int(best_sat[ue])
            if b < 0:
                # No visible satellite
                serving[ue] = -1
                ho_timer[ue] = 0
                curr_metric_db[ue] = -1e9
                # Log outage start
                if s_old >= 0:
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "outage_start",
                        "from": int(s_old), "to": -1,
                        "prev_metric_db": met_old,
                    })
                continue
            if serving[ue] < 0:
                serving[ue] = b
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                ho_events[ue].append({
                    "t": int(t_idx), "type": "attach",
                    "from": -1, "to": int(b),
                    "metric_db": float(best_metric_db[ue]),
                })
                continue
            if b == serving[ue]:
                # Same serving; refresh metric
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                # If continuing after outage end
                if s_old < 0 and serving[ue] >= 0:
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "outage_end",
                        "from": -1, "to": int(serving[ue]),
                        "metric_db": float(best_metric_db[ue]),
                    })
                continue
            # Candidate different than serving
            if not ho_enabled:
                serving[ue] = b
                curr_metric_db[ue] = best_metric_db[ue]
                ho_timer[ue] = 0
                ho_events[ue].append({
                    "t": int(t_idx), "type": "handover",
                    "from": int(s_old), "to": int(b),
                    "prev_metric_db": met_old,
                    "metric_db": float(best_metric_db[ue]),
                })
                continue
            diff_db = best_metric_db[ue] - curr_metric_db[ue]
            if diff_db > ho_hyst_db:
                ho_timer[ue] += 1
                if ho_timer[ue] >= ho_ttt:
                    serving[ue] = b
                    curr_metric_db[ue] = best_metric_db[ue]
                    ho_timer[ue] = 0
                    ho_events[ue].append({
                        "t": int(t_idx), "type": "handover",
                        "from": int(s_old), "to": int(b),
                        "prev_metric_db": met_old,
                        "metric_db": float(best_metric_db[ue]),
                        "diff_db": float(diff_db),
                        "hyst_db": float(ho_hyst_db),
                        "ttt_ttis": int(ho_ttt),
                    })
            else:
                ho_timer[ue] = 0

        # Outage accumulation and optional serving trace
        outage_ttis += (serving < 0).astype(int)
        if include_trace and serving_trace is not None:
            serving_trace.append(np.array(serving, copy=True))

        # Build per-satellite UE subsets and schedule independently
        mcs_params = {
            "olla_offset_db": config.get("csi_olla_offset_db", 0.0),
            "mcs_table": config.get("csi_mcs_table", "legacy"),
            "residual_freq_hz": config.get("residual_freq_hz", 0.0),
            "scs_khz": config.get("scs_khz", 30),
        }
        per_sat_served_counts: Dict[int, int] = {}
        for si in cand:
            ue_idx = np.flatnonzero(serving == si)
            if ue_idx.size == 0:
                continue
            per_sat_served_counts[si] = int(ue_idx.size)
            # Use full UE arrays; apply scheduling mask to restrict served UEs for this satellite
            cap = cap_s[si]
            snr_lin = snr_lin_s[si]
            cap_wb = np.log2(1.0 + np.maximum(snr_wb_s[si], 1e-12))
            snr_wb = snr_wb_s[si]
            mask = np.zeros(N_UE, dtype=bool)
            mask[ue_idx] = True
            ue_mask = mask[None, :]

            # One-TTI Baseline-Default: contiguous-block PF with baseline per-PRB metric
            # Fixed: use CQI quantization to form baseline metric
            se_base_prb = snr_to_se_sched(
                snr_lin, config.get("use_mcs", False), mcs_params,
                enable_cqi_quant=True,
                cqi_table=config.get("csi_mcs_table", "nr_256qam")
            )
            # Prepare per-satellite HARQ managers if enabled
            harq_base = harq_base_by_sat.get(si)
            harq_rm = harq_rm_by_sat.get(si)
            if harq_base is None and bool(config.get("enable_harq_full", False)):
                harq_base = HarqManagerFull(
                    num_ue=N_UE,
                    num_procs=int(config.get("harq_max_procs", 16)),
                    ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                    config=config,
                )
                harq_base_by_sat[si] = harq_base
            if harq_rm is None and bool(config.get("enable_harq_full", False)):
                harq_rm = HarqManagerFull(
                    num_ue=N_UE,
                    num_procs=int(config.get("harq_max_procs", 16)),
                    ack_delay_ttis=int(config.get("harq_ack_delay_ttis", 10)),
                    config=config,
                )
                harq_rm_by_sat[si] = harq_rm

            # Disable tail flush in streaming mode (we call per TTI)
            _cfg_base = dict(config); _cfg_base['harq_flush_tail'] = False
            _cfg_base["scheduler_kind"] = "heuristic"
            _cfg_base["nsgbs_model_path"] = None
            _cfg_base["nsgbs_collect_stats"] = False
            _cfg_base["nsgbs_stats_out"] = None
            _cfg_base["nsgbs_collect_dataset"] = False
            _cfg_base["nsgbs_dataset_out"] = None
            base = pf_schedule_radiomap_blocks(
                cap, 1, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=prb_cap_unified,
                mcs_params=mcs_params,
                se_metric_time=se_base_prb[None, ...],
                snr_lin_time=None,
                eesm_beta_db=float(config.get("baseline_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=ue_mask,
                harq_mgr=harq_base,
                dl_power_model=dlpm,
                P_tot_dbm=Ptot,
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=pmin,
                p_max_dbm=pmax,
                record_assignments=False,
                assignments_out=None,
                record_ue_thr=False,
                ue_thr_out=None,
                config=_cfg_base,
            )
            _cfg_rm = dict(config); _cfg_rm['harq_flush_tail'] = False
            rm = pf_schedule_radiomap_blocks(
                cap, 1, beta=config["pf_beta"],
                snr_lin=snr_lin,
                overhead_eff=config.get("overhead_eff", 1.0),
                use_mcs=config.get("use_mcs", False),
                power_split=config.get("power_split", False),
                se_metric_override=None,
                max_prbs_per_ue=prb_cap_unified,
                mcs_params=mcs_params,
                se_metric_time=None,
                snr_lin_time=None,
                eesm_beta_db=float(config.get("rm_sched_eesm_beta_db", config.get("sched_eesm_beta_db", 1.0))),
                require_contiguous=bool(config.get("sched_require_contiguous", True)),
                rng=rng,
                ue_mask_time=ue_mask,
                harq_mgr=harq_rm,
                dl_power_model=dlpm,
                P_tot_dbm=Ptot,
                P_ref_dbm=config.get("P_tx_dbm"),
                p_min_dbm=pmin,
                p_max_dbm=pmax,
                record_assignments=False,
                assignments_out=None,
                record_ue_thr=False,
                ue_thr_out=None,
                config=_cfg_rm,
            )
            sum_rate_base_def += base * Z
            sum_rate_rm += rm * Z
            # Update per-satellite KPI
            k = kpi_per_sat.get(si)
            if k is None:
                k = {
                    "name": getattr(orbit.sats[si], 'name', f"SAT-{int(si)}"),
                    "ttis_active": 0,
                    "served_ue_sum": 0,
                    "served_ue_max": 0,
                    "sum_se_base_def": 0.0,
                    "sum_se_rm": 0.0,
                }
                kpi_per_sat[si] = k
            k["ttis_active"] += 1
            k["served_ue_sum"] += int(ue_idx.size)
            k["served_ue_max"] = max(int(k["served_ue_max"]), int(ue_idx.size))
            k["sum_se_base_def"] += float(base * Z)
            k["sum_se_rm"] += float(rm * Z)

        # Initialize kpi dict if first time
        # (Declared before loop to satisfy type checker.)
        # Per-satellite no-UE case: not counted as active.

    # Average across T and PRBs (per original convention): divide by T and Z
    avg_se_base_def = sum_rate_base_def / max(1, T) / max(1, Z)
    avg_se_rm = sum_rate_rm / max(1, T) / max(1, Z)

    # Summarize per-satellite KPI
    per_sat_summary = []
    for si, k in sorted(kpi_per_sat.items(), key=lambda x: x[0]):
        tt = int(k["ttis_active"]) or 1
        per_sat_summary.append({
            "sat_index": int(si),
            "name": str(k["name"]),
            "ttis_active": int(k["ttis_active"]),
            "avg_served_ue": float(k["served_ue_sum"]) / float(tt),
            "max_served_ue": int(k["served_ue_max"]),
            "avg_se_base_default_per_prb": float(k["sum_se_base_def"]) / float(tt * max(1, Z)),
            "avg_se_rm_per_prb": float(k["sum_se_rm"]) / float(tt * max(1, Z)),
        })

    imp_pct = (avg_se_rm - avg_se_base_def) / max(1e-9, avg_se_base_def) * 100.0 if avg_se_base_def > 0 else float('inf')

    # Aggregate HARQ stats across satellites (concise summary)
    def _aggregate_harq(hdict: Dict[int, HarqManagerFull]):
        if not hdict:
            return None
        total_started = 0
        total_acked = 0
        total_dropped = 0
        total_init_ack = 0
        total_retx_weighted = 0.0
        for si, mgr in hdict.items():
            if mgr is None or not hasattr(mgr, 'get_stats'):
                continue
            hs = mgr.get_stats()
            tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
            tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
            tb_dropped = int(hs.get('tb_dropped', 0))
            init_ack = int(hs.get('initial_ack_count', 0))
            avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
            total_started += tb_started
            total_acked += tb_acked
            total_dropped += tb_dropped
            total_init_ack += init_ack
            # Weighted by acked TBs
            total_retx_weighted += avg_retx * max(0, tb_acked)
        if total_started <= 0:
            return None
        ack_rate = total_acked / float(total_started)
        first_try = total_init_ack / float(total_started)
        avg_retx = (total_retx_weighted / float(total_acked)) if total_acked > 0 else 0.0
        return {
            'tb_started': int(total_started),
            'tb_acked': int(total_acked),
            'tb_dropped': int(total_dropped),
            'ack_rate': float(ack_rate),
            'first_try_ack_rate': float(first_try),
            'avg_retx_per_acked': float(avg_retx),
        }

    harq_stats_base = _aggregate_harq(harq_base_by_sat)
    harq_stats_map = _aggregate_harq(harq_rm_by_sat)

    report = {
        "avg_se_baseline_default": avg_se_base_def,
        "avg_se_radiomap": avg_se_rm,
        "improvement_vs_default_pct": imp_pct,
        "R_xyz_dbm": R_xyz_dbm,
        "ue_pos": ue_pos,
        "T": int(T),
        "Z": int(Z),
        "N_UE": int(N_UE),
        # Bandwidth/throughput metrics
        "prb_bw_hz": float(prb_bw_hz),
        "system_bandwidth_hz": float(prb_bw_hz) * float(Z),
        "total_throughput_baseline_bps": float(avg_se_base_def * prb_bw_hz * Z),
        "total_throughput_radiomap_bps": float(avg_se_rm * prb_bw_hz * Z),
        "avg_ue_throughput_baseline_bps": float((avg_se_base_def * prb_bw_hz * Z) / max(1, int(N_UE))),
        "avg_ue_throughput_radiomap_bps": float((avg_se_rm * prb_bw_hz * Z) / max(1, int(N_UE))),
        "ho_events_per_ue": ho_events,
        "handover_count_per_ue": [int(sum(1 for e in ho_events[i] if e.get("type") == "handover")) for i in range(N_UE)],
        "outage_ttis_per_ue": outage_ttis.tolist(),
        "per_sat_kpis": per_sat_summary,
        "sat_index_to_name": {int(i): getattr(orbit.sats[i], 'name', f"SAT-{int(i)}") for i in range(len(orbit.sats))},
        "harq_stats_base": harq_stats_base,
        "harq_stats_map": harq_stats_map,
    }
    if include_trace and serving_trace is not None:
        report["serving_trace"] = np.stack(serving_trace, axis=0)

    # Optional JSON report
    try:
        if bool(config.get("write_json_report", False)):
            out_dir = config.get("plot_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            name = str(config.get("report_basename", "constellation_summary"))
            path = os.path.join(out_dir, f"{name}.json")
            def serialize(obj):
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                raise TypeError
            with open(path, 'w') as f:
                import json as _json
                _json.dump(report, f, default=serialize)
    except Exception as e:
        print(f"[WARN] Constellation JSON report failed: {e}")

    return report

# CONFIG is provided by code/config.py

if __name__ == '__main__':
    # -----------------------
    # Run constellation or single-satellite experiment
    # -----------------------
    if bool(CONFIG.get("enable_constellation", False)):
        out = run_constellation(CONFIG)
        print("Constellation-run results (independent scheduling, no inter-sat interference)")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {out['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap         avg SE (bits/s/Hz): {out['avg_se_radiomap']:.3f}")
        try:
            print(f"  Gain vs Default (%): {out['improvement_vs_default_pct']:.2f}")
        except Exception:
            pass
        # Optional concise HARQ summary (constellation aggregate)
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq_const(label: str, hs: dict) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', 0))
                tb_acked = int(hs.get('tb_acked', 0))
                tb_dropped = int(hs.get('tb_dropped', 0))
                ack_rate = float(hs.get('ack_rate', 0.0)) * 100.0
                first_try = float(hs.get('first_try_ack_rate', 0.0)) * 100.0
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                print(f"\n[HARQ] {label} (aggregate):")
                print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={first_try:.1f}%)")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
            try:
                _print_harq_const("Baseline-Default", out.get("harq_stats_base"))
                _print_harq_const("RadioMap", out.get("harq_stats_map"))
            except Exception:
                pass
    else:
        single = run_once(CONFIG)
        print("Single-run results")
        print(f"  Baseline-Default avg SE (bits/s/Hz): {single['avg_se_baseline_default']:.3f}")
        print(f"  RadioMap        avg SE (bits/s/Hz): {single['avg_se_radiomap']:.3f}")
        print(f"  Gain vs Default (%): {single['improvement_vs_default_pct']:.2f}")
        try:
            bw_mhz = single.get('system_bandwidth_hz', 0.0) / 1e6
            th_base = single.get('total_throughput_baseline_bps', None)
            th_map = single.get('total_throughput_radiomap_bps', None)
            if th_base is not None and th_map is not None:
                print(f"  System Bandwidth: {bw_mhz:.3f} MHz")
                print(f"  Baseline-Default total throughput: {th_base/1e6:.3f} Mbps")
                print(f"  RadioMap        total throughput: {th_map/1e6:.3f} Mbps")
                if 'avg_ue_throughput_baseline_bps' in single and 'avg_ue_throughput_radiomap_bps' in single:
                    print(f"  Avg UE throughput (Baseline): {single['avg_ue_throughput_baseline_bps']/1e6:.3f} Mbps/UE")
                    print(f"  Avg UE throughput (RadioMap): {single['avg_ue_throughput_radiomap_bps']/1e6:.3f} Mbps/UE")
        except Exception:
            pass

        # Optional concise HARQ summary
        if bool(CONFIG.get("print_harq_summary", True)):
            def _print_harq(label: str, hs: dict, per_ue_key: str) -> None:
                if not hs:
                    print(f"\n[HARQ] {label}: no HARQ stats available.")
                    return
                tb_started = int(hs.get('tb_started', hs.get('initial_ack_count', 0) + hs.get('initial_nack_count', 0)))
                tb_acked = int(hs.get('tb_acked', hs.get('ack_count', 0)))
                tb_dropped = int(hs.get('tb_dropped', 0))
                init_ack = int(hs.get('initial_ack_count', 0))
                avg_retx = float(hs.get('avg_retx_per_acked', 0.0))
                olla_hist = hs.get('olla_offset_avg', []) or []
                olla_last = float(olla_hist[-1]) if len(olla_hist) > 0 else float(np.mean(hs.get('olla_last_per_ue', []) or [0.0]))
                print(f"\n[HARQ] {label}:")
                if tb_started > 0:
                    ack_rate = 100.0 * tb_acked / float(tb_started)
                    init_ack_rate = 100.0 * init_ack / float(tb_started)
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}  (ACK rate={ack_rate:.1f}%, first-try ACK={init_ack_rate:.1f}%)")
                else:
                    print(f"  TB started/ACKed/dropped: {tb_started}/{tb_acked}/{tb_dropped}")
                print(f"  Avg retransmissions per ACKed TB: {avg_retx:.2f}")
                print(f"  OLLA avg offset (last): {olla_last:+.2f} dB")
                # Per-UE goodput (SE per PRB) if available
                per_ue = single.get(per_ue_key)
                if per_ue:
                    arr = np.asarray(per_ue, dtype=float)
                    print(f"  Per-UE avg SE: mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

            _print_harq("Baseline-Default", single.get("harq_stats_base"), "per_ue_avg_se_base")
            _print_harq("RadioMap", single.get("harq_stats_map"), "per_ue_avg_se_map")

        # -----------------------
        # Run multiple seeds to show robustness
        # -----------------------
        seeds = np.arange(1, 21)
        multi = run_many(CONFIG, seeds)
        print("\nMulti-seed summary (N=20)")
        print(f"  Baseline-Default avg SE: {multi['baseline_default'].mean():.3f} ± {multi['baseline_default'].std():.3f}")
        print(f"  RadioMap         avg SE: {multi['radiomap'].mean():.3f} ± {multi['radiomap'].std():.3f}")
        print(f"  Gain vs Default median: {np.median(multi['improvement_vs_default_pct']):.2f}% (min={multi['improvement_vs_default_pct'].min():.2f}%, max={multi['improvement_vs_default_pct'].max():.2f}%)")

        # -----------------------
        # Plots
        # -----------------------
        save_plots = CONFIG.get("save_plots", True)
        show_plots = CONFIG.get("show_plots", False)
        plot_dir = CONFIG.get("plot_dir", "output")
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        def maybe_finalize(fig_name: str):
            if save_plots:
                plt.savefig(os.path.join(plot_dir, fig_name), dpi=140, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

        # 1) Improvement distribution
        plt.figure(figsize=(6,4))
        plt.hist(multi["improvement_vs_default_pct"], bins=10, edgecolor='black')
        plt.title("Radio Map–aware gain vs Default baseline")
        plt.xlabel("Gain vs. Default baseline (%)")
        plt.ylabel("Count")
        plt.tight_layout()
        maybe_finalize("gain_distribution.png")

        # 2) Example Interference Map slice (median over frequency)
        R_med = np.median(single["R_xyz_dbm"], axis=2)
        plt.figure(figsize=(5,5))
        plt.imshow(R_med.T, origin='lower', aspect='equal')
        plt.title("Interference Map (median over frequency), dBm")
        plt.colorbar(label='dBm')
        plt.tight_layout()
        maybe_finalize("interference_map_median.png")

        # 3) Example per-UE wideband vs best-PRB capacity (first 10 UEs)
        ue = np.arange(min(10, CONFIG["N_UE"]))
        best_prb = single["cap"][ue].max(axis=1)
        wb = single["cap_wb"][ue]
        x = np.arange(ue.size)
        plt.figure(figsize=(6,4))
        plt.bar(x - 0.2, wb, width=0.4, label='Wideband (baseline)')
        plt.bar(x + 0.2, best_prb, width=0.4, label='Best PRB (RadioMap)')
        plt.xticks(x, [f"UE{int(i)}" for i in ue])
        plt.ylabel("Spectral efficiency (bits/s/Hz)")
        plt.title("Per-UE: wideband vs best PRB opportunity")
        plt.legend()
        plt.tight_layout()
        maybe_finalize("per_ue_wb_vs_best_prb.png")
