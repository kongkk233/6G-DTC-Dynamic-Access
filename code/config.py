"""
Central configuration for the NR-NTN downlink (DL) simulation.

This repo is DL‑only. All UL‑specific features (UL power control, TA, UL
Doppler pre‑compensation) and their configs are removed. The Radio Map models
terrestrial interference at the UE receiver per PRB.
"""

import os


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    # --- Core simulation ---
    "Z": 51,                 # PRBs (must match Radio Map Z)
    "N_UE": 100,             # active UEs per run
    "T": 2000,               # TTIs per run
    "seed": 101,             # RNG seed

    # --- Radio Map input ---
    "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
    "radio_map_mat_var": "XdB_recon_tensor",
    "radio_map_units": "dBm",  # one of {"dBm", "mW", "W"}

    # --- Noise / numerology ---
    "scs_khz": 30,           # PRB BW = 12*30 kHz = 360 kHz
    "cp_type": "normal",     # FR1 CP
    "noise_temp_K": 290.0,

    # --- Channel model (3GPP NTN) ---
    "channel_model": "3gpp_ntn",
    "ntn_channel_profile": "s_band_handheld_urban",
    # Optional overrides to the profile
    "channel_params": {
        "additional_loss_db": {"slos": 7.0, "nlos": 16.0},
        "shadow_sigma_db": {"slos": 4.0, "nlos": 6.0},
        "k_factor_db": {"los": 14.0, "slos": 8.0},
    },

    # --- Link budget ---
    "P_tx_dbm": 30.0,        # DL per‑PRB EIRP baseline (equal-power)
    "G_rx_db": 38.0,         # beam boresight gain
    "shadow_std_db": 7.0,    # lognormal shadowing std (dB)
    "rx_nf_db": 7.0,         # UE noise figure (dB)
    "impl_loss_db": 1.0,     # implementation loss as noise rise (dB)
    "overhead_eff": 0.85,    # PHY/MAC overhead factor
    "pf_beta": 0.1,          # PF averaging factor

    # --- Scheduler realism ---
    "use_mcs": True,         # map SNR->SE via MCS table
    "csi_olla_offset_db": 0.0,   # OLLA offset (dB)
    # Baseline uses CQI quantization (fixed in code)
    "baseline_csi_delay_ttis": 12,  # baseline CSI delay (TTIs)
    "rm_csi_delay_ttis": 0,         # RadioMap CSI delay (TTIs)
    "power_split": False,           # per-UE power split penalty

    # --- CSI periodicity / RM estimation ---
    "enable_cqi_periodicity_base": True,
    "enable_cqi_periodicity_rm": False,
    "cqi_period_ttis": 5,
    "cqi_offset_ttis": 0,
    "radiomap_est_error_db": 1.5,
    "radiomap_blur_sigma": 1.0,

    # --- Geometry / beam ---
    "sat_altitude_km": 600.0,
    "carrier_freq_GHz": 2.000,
    "beam_center_xy": None,      # default: map center
    "beam_half_bw_deg": 8.0,
    "beam_edge_drop_db": 3.0,
    "cell_size_km": 0.125,       # ground resolution per pixel

    # --- Orbit dynamics ---
    "enable_orbit_dynamics": True,
    "tti_ms": 1.0,
    "sat_ground_speed_kms": 7.5,
    "sat_heading_deg": 0.0,

    # --- Radio Map dynamics ---
    "enable_time_varying": True,
    "rm_flicker_db_std": 0.5,
    # Flicker dimensionality:
    # - "global": one wideband offset shared by all (x,y,z) each TTI (default; avoids artificial PRB diversity)
    # - "pixel": one wideband offset per (x,y) each TTI, shared across PRBs
    # - "prb": one offset per PRB each TTI, shared across space
    # - "element": i.i.d. per (x,y,z) each TTI (can overstate exploitable diversity)
    "rm_flicker_kind": "global",
    # Flicker distribution:
    # - "rectified" (default): add extra interference only (never below baseline)
    # - "signed": symmetric N(0, std) in dB (can lead to counterintuitive SE increases)
    # - "abs": |N(0, std)| in dB (always >0, stronger than rectified)
    "rm_flicker_dist": "rectified",
    "rm_drift_px": (0, 0),
}

# Scheduler
CONFIG.update({
    "sched_require_contiguous": True,  # one contiguous block per UE per TTI
    # Default EESM beta (fallback)
    "sched_eesm_beta_db": 2.5,
    # Per-path overrides (slightly higher to reduce freq-selective penalty)
    # Baseline tuned a bit more conservative than RM to stabilize ACKs
    "baseline_sched_eesm_beta_db": 2.7,
    "rm_sched_eesm_beta_db": 3.5,
    # Scheduler kind: "heuristic" (default) or "nsgbs" (neural-scored greedy block scheduler)
    "scheduler_kind": "heuristic",
    # NS-GBS inference knobs (Phase1 scaffolding; model integration added later)
    "nsgbs_model_path": None,
    "nsgbs_topB": 4,
    "nsgbs_window": 3,
    "nsgbs_use_harq_features": True,
    "nsgbs_score_mode": "classify",
    "nsgbs_add_z": False,
    "nsgbs_add_step": False,
    "nsgbs_collect_stats": False,
    "nsgbs_stats_out": None,
    # Dataset collection for NS-GBS (Phase2)
    "nsgbs_collect_dataset": False,
    "nsgbs_collect_stride": 1,
    "nsgbs_collect_max_samples": None,
    # NS-GBS inference device (optional)
    "nsgbs_device": None,
})

# Access gating/HO removed in minimal DL-only preset

# HARQ/BLER/OLLA (optional; generic to DL)
CONFIG.update({
    # Always use full HARQ in this build
    "enable_harq_deferral": False,
    "harq_max_procs": 24,
    "harq_ack_delay_ttis": 6,
    "enable_harq_full": True,
    "harq_target_bler": 0.1,
    "harq_max_retx": 4,
    # 3GPP Table 2 (256QAM-capable)
    "mcs_table_kind": "3gpp_table_2",
    # PDSCH DMRS/overhead for N_RE computation
    "pdsch_dmrs_sym_per_slot": 1,
    "dmrs_re_per_sym_per_prb": 6,
    "oh_prb": 0,
    # BLER curve parameters (AWGN-like sigmoid)
    "bler_slope_db": 1.0,
    "bler_margin_db": 1.5,
    # Optional external BLER curves JSON (per table/MCS idx). If set, overrides AWGN model.
    "bler_curve_path": None,
    # OLLA steps (slightly stronger initial conservatism & steps)
    "olla_step_up_db": 0.06,
    "olla_step_down_db": 0.12,
    "olla_init_offset_db": -1.5,
    "olla_min_db": -3.0,
    "olla_max_db": 6.0,
    # Retransmission scheduling priority boost in PF metric (additive)
    "harq_retx_priority_bonus": 0.5,
    # Tail-ACK flush after last TTI to remove boundary loss
    "harq_flush_tail": True,
    # Optional path to 3GPP MCS tables (JSON). If set, you can choose '3gpp_table_1/2/3'.
    "mcs_3gpp_table_path": os.path.join(_BASE_DIR, "..", "docs", "mcs_tables_38_214.json"),
    # 64QAM (Table 1) - 改用3GPP标准JSON表
    "csi_mcs_table": "3gpp_table_1",
})

# Multi-beam and external orbit models removed in minimal preset

# A3/HO parameters removed

# Skyfield/TLE-driven orbit (required when dynamics are enabled)
CONFIG.update({
    # Provide either two-line TLE via 'tle_lines' (list[str,str]) or a file path via 'tle_path'
    # "tle_name": "STARLINK-11087 [DTC]",
    # "tle_lines": [
    #     "1 59421C 24065A   25259.10395833  .00000071  00000+0  58945-6 0  2596",
    #     "2 59421  53.1566 234.6672 0001137  93.5671 155.3063 15.69664283    16",
    # ],
    "tle_name": "STARLINK-11090 [DTC]",
    "tle_lines": [
        "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
        "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
    ],
    "tle_path": None,
    # Orbit start time for t=0 (ISO8601).
    # Updated to the peak-elevation overpass over Shanghai (see tools/find_overpass_times.py)
    # "orbit_start_datetime": "2025-09-13T19:01:17.393469Z",
    "orbit_start_datetime": "2025-10-07T21:38:31.574982+00:00",
    # Mapping the simulation grid (x,y) to Earth surface around a reference lat/lon (degrees).
    # Each pixel corresponds to 'cell_size_km' in local ENU, with an optional rotation.
    # Anchor the ground map at Shanghai city center
    "auto_ref_from_tle": False,
    # "ref_lat_deg": 31.2304,
    # "ref_lon_deg": 121.4737,
    "ref_lat_deg": 43.65108,
    "ref_lon_deg": -79.34702,
    "map_rotation_deg": 0.0,
})

# Printing helpers
CONFIG.update({
    # Print concise HARQ summary after single run
    "print_harq_summary": True,
    # RadioMap-specific DL power allocation: enable water-filling with total-power constraint
    "rm_dl_power_model": "waterfill",
    # Power allocation knobs (≈ 33 dBm/PRB * 51 PRBs ≈ 50 dBm total)
    "rm_P_tot_dbm": 50.0,
    # Allow more concentration on high-SINR PRBs while avoiding starvation
    "rm_p_min_dbm": 28.0,
    "rm_p_max_dbm": 36.0,
    # Baseline now uses moderated water-filling (tighter bounds vs RM)
    "baseline_dl_power_model": "waterfill",
    # Match equal-power budget to 33 dBm/PRB * 51 PRBs ≈ 50 dBm
    "baseline_P_tot_dbm": 50.0,
    # Allow slightly wider per-PRB box constraints for better fit
    "baseline_p_min_dbm": 27.0,
    "baseline_p_max_dbm": 33.0,
    # Per-path PRB caps
    "baseline_max_prbs_per_ue": 20,
    "rm_max_prbs_per_ue": 20,
})

# Constellation (multi-satellite) options
CONFIG.update({
    # Enable constellation mode by default for DTC evaluation
    "enable_constellation": False,
    # Path to DTC constellation TLE catalog
    # Switched to filtered Satnet (590 km) catalog to match current experiment
    "tle_catalog_path": os.path.join(_BASE_DIR, "..", "tles", "Satnet_DTC.txt"),
    # Candidate filtering near the ground map reference (km)
    "constellation_max_ground_radius_km": 1200.0,
    # Limit the number of satellites considered per TTI (after filtering)
    "constellation_max_sats_per_tti": 6,
    # Minimum UE elevation (deg) for visibility/association
    "min_elev_deg": 20.0,
    # Association metric: 'snr_wb' (linear), 'prx_dbm'
    "association_metric": "snr_wb",
    # Handover control (constellation mode)
    "ho_enabled": True,
    "ho_hyst_db": 2.0,
    "ho_ttt_ttis": 20,
    # Reporting
    "write_json_report": True,
    "report_basename": "constellation_summary",
    # Whether to include full serving timeline [T, N_UE] in JSON (may be large)
    "include_serving_trace": False,
    # Unify per-satellite PRB cap in constellation mode for fair comparison
    "constellation_prb_cap": 20,
})
