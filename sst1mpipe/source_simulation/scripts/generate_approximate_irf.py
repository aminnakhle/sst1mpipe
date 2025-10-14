#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate approximate IRF CSV files for SST-1M from DL2 data (gamma + proton),
keeping the LST simulator CSV formats.

Outputs (per cut efficiency pair):
  - Gamma CSV columns: ZD_deg,Etrue_min_TeV,Etrue_max_TeV,Aeff_m2,emig_mu_loc,emig_mu_scale,emig_mu_a,emig_model
  - Background CSV columns: ZD_deg,Ereco_min_TeV,Ereco_max_TeV,BckgRate_per_second,Theta_cut_deg,Gammaness_cut

Key points:
  * Cuts are computed PER TRUE-ENERGY BIN using pyirf:
        calculate_percentile_cut + evaluate_binned_cut
  * Aeff is computed with pyirf.effective_area_per_energy
  * Energy dispersion parameters are fitted per true-energy bin on selected gamma events.
    (We also compute pyirf.energy_dispersion for inspection if --save-edisp-matrix is set.)
  * Background:
      - default: shape-only spectrum from proton MC after cuts, scaled to --bkg-total-rate [Hz]
      - optional: physically-anchored CR power-law weighting with --cr-powerlaw

Paths (Windows-friendly defaults):
  --dl2-dir  C:\\Users\\a.nakhle\\sst1mpipe\\sst1mpipe\\source_simulation\\dl2_gamma
  --output-dir C:\\Users\\a.nakhle\\sst1mpipe\\sst1mpipe\\source_simulation\\SST1M_csv
  --lst-ref-dir C:\\Users\\a.nakhle\\sst1mpipe\\sst1mpipe\\source_simulation\\lst_data  (not required; only for manual comparisons)

Example runs:
  # single pair of cuts
  python generate_approximate_irf.py --gh-cut 0.70 --th-cut 0.70 --zenith 20

  # grid: 0.40, 0.70, 0.90 (gh=th each)
  python generate_approximate_irf.py --grid 0.40,0.70,0.90

  # use CR weighting for background and save edisp matrices for debugging
  python generate_approximate_irf.py --gh-cut 0.70 --th-cut 0.70 --cr-powerlaw --cr-k 1.8e-7 --cr-index 2.7 --save-edisp-matrix
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import glob
import warnings
import inspect

import astropy.units as u
from astropy.table import Table, vstack

from scipy.stats import moyal, skewnorm

# Fix Windows compatibility issue with ctapipe (pwd module)
if sys.platform == 'win32':
    import types
    sys.modules['pwd'] = types.ModuleType('pwd')
    sys.modules['pwd'].getpwuid = lambda x: types.SimpleNamespace(pw_name='user')

from ctapipe.io import read_table

# --- pyirf imports per LST expert advice ---
from pyirf.irf import energy_dispersion, effective_area_per_energy
from pyirf.simulations import SimulatedEventsInfo
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.binning import create_bins_per_decade


# --------------------------- Helpers --------------------------------
def _to_value_tev(arr):
    """Return a numpy array of energies in TeV (handles Quantity or plain floats)."""
    if hasattr(arr, 'unit'):
        # For astropy Column/Quantity, use .quantity property then to_value
        if hasattr(arr, 'quantity'):
            return np.asarray(arr.quantity.to_value(u.TeV))
        else:
            return np.asarray(arr.to_value(u.TeV))
    return np.asarray(arr, dtype=float)

def _to_value_rad(arr):
    if hasattr(arr, 'unit'):
        # For astropy Column/Quantity, use .quantity property then to_value
        if hasattr(arr, 'quantity'):
            qty = arr.quantity
            # Check if dimensionless (no unit) - assume already in radians
            if qty.unit == u.dimensionless_unscaled or str(qty.unit) == '':
                return np.asarray(qty.value)
            return np.asarray(qty.to_value(u.rad))
        else:
            if arr.unit == u.dimensionless_unscaled or str(arr.unit) == '':
                return np.asarray(arr.value)
            return np.asarray(arr.to_value(u.rad))
    return np.asarray(arr, dtype=float)

def _theta_from_altaz(true_alt, true_az, reco_alt, reco_az):
    """
    Angular separation (rad) between true and reco directions (haversine on sphere).
    Inputs can be quantities or floats; assumed radians if floats.
    """
    talt = _to_value_rad(true_alt)
    taz  = _to_value_rad(true_az)
    ralt = _to_value_rad(reco_alt)
    raz  = _to_value_rad(reco_az)

    cos_theta = (np.sin(talt) * np.sin(ralt)
                 + np.cos(talt) * np.cos(ralt) * np.cos(taz - raz))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # radians

def _safe_percentile_cut(values, bin_values, bins, eff):
    """
    Robust wrapper to call calculate_percentile_cut with either `percentile` or `efficiency`.
    
    - 'efficiency' keyword expects [0..1]
    - 'percentile' keyword expects [0..100]
    
    Falls back to numpy if signature mismatch occurs.
    """
    v = np.asarray(values, dtype=float)
    bv = np.asarray(bin_values, dtype=float)
    edges = np.asarray(bins, dtype=float)

    try:
        # Check which parameter the function accepts
        sig = inspect.signature(calculate_percentile_cut)
        if 'efficiency' in sig.parameters:
            # efficiency expects [0..1]
            return calculate_percentile_cut(values=v, bins=edges, bin_values=bv, efficiency=float(eff))
        elif 'percentile' in sig.parameters:
            # percentile expects [0..100]
            return calculate_percentile_cut(values=v, bins=edges, bin_values=bv, percentile=float(eff) * 100.0)
        else:
            raise TypeError("Unknown parameter name")
    except Exception:
        pass

    # Fallback: compute per-bin percentiles with numpy (expects 0..100)
    cut = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (bv >= edges[i]) & (bv < edges[i+1])
        if np.any(m):
            cut[i] = np.percentile(v[m], eff * 100.0)
    return cut

def _eval_binned(values, bin_values, bins, cut, op):
    """
    Robust wrapper for evaluate_binned_cut with fallback.
      op: ">=" or "<="
    """
    v  = np.asarray(values, dtype=float)
    bv = np.asarray(bin_values, dtype=float)
    edges = np.asarray(bins, dtype=float)

    try:
        return evaluate_binned_cut(values=v, bins=edges, bin_values=bv, cut=cut, operator=op)
    except Exception:
        # Fallback: manual evaluation
        sel = np.zeros_like(v, dtype=bool)
        idx = np.digitize(bv, edges) - 1
        good = (idx >= 0) & (idx < len(cut))
        if op == ">=":
            sel[good] = v[good] >= cut[idx[good]]
        else:
            sel[good] = v[good] <= cut[idx[good]]
        return sel

def _fit_residuals_per_true_bin(true_e, reco_e, selected_mask, e_true_edges):
    """
    For each true-energy bin, fit residuals ( (Ereco-Etrue)/Etrue ) with moyal
    and optionally skewnorm. Returns list of dicts with mu_loc, mu_scale, mu_a, model.
    """
    te  = np.asarray(true_e, dtype=float)
    re  = np.asarray(reco_e, dtype=float)
    sel = np.asarray(selected_mask, dtype=bool)

    n_bins = len(e_true_edges) - 1
    out = []
    for i in range(n_bins):
        m = sel & (te >= e_true_edges[i]) & (te < e_true_edges[i+1])
        if m.sum() < 20:
            out.append(dict(mu_loc=np.nan, mu_scale=np.nan, mu_a=np.nan, model="moyal"))
            continue

        rel = (re[m] - te[m]) / np.clip(te[m], 1e-12, None)

        # robust outlier removal
        med = np.median(rel)
        mad = np.median(np.abs(rel - med))
        sigma = max(1.4826 * mad, 1e-3)
        clean = rel[np.abs(rel - med) < 3.0 * sigma]
        if clean.size < 20:
            clean = rel  # keep original if too small

        # Try moyal first
        try:
            loc_m, scale_m = moyal.fit(clean, loc=med, scale=sigma)
        except Exception:
            loc_m, scale_m = med, sigma

        # Sanity clamp
        ok_m = np.isfinite(loc_m) and np.isfinite(scale_m) and (abs(loc_m) < 2.0) and (0.005 < scale_m < 3.0)

        # Try skewnorm
        try:
            a_s, loc_s, scale_s = skewnorm.fit(clean, loc=med, scale=sigma)
            ok_s = (np.isfinite(a_s) and np.isfinite(loc_s) and np.isfinite(scale_s)
                    and abs(a_s) < 30 and abs(loc_s) < 2.0 and 0.005 < scale_s < 3.0)
        except Exception:
            ok_s = False
            a_s = loc_s = scale_s = np.nan

        if ok_s and (abs(a_s) > 0.5):
            out.append(dict(mu_loc=float(loc_s), mu_scale=float(scale_s), mu_a=float(a_s), model="skewnorm"))
        elif ok_m:
            out.append(dict(mu_loc=float(loc_m), mu_scale=float(scale_m), mu_a=np.nan, model="moyal"))
        else:
            out.append(dict(mu_loc=float(med), mu_scale=float(sigma), mu_a=np.nan, model="moyal"))

    return out

def _solid_angle_from_viewcone(viewcone_deg):
    """Return solid angle [sr] for a symmetric viewcone (0..180 deg)."""
    v = float(max(viewcone_deg, 0.0))
    return 2.0 * np.pi * (1.0 - np.cos(np.deg2rad(v)))

def _powerlaw_flux_integral(Emin_TeV, Emax_TeV, K, index):
    """
    ∫ K * E^{-index} dE from Emin..Emax (TeV-based units).
    Returns K/(1-index) * (Emax^{1-index} - Emin^{1-index}) for index != 1.
    """
    Emin = float(Emin_TeV)
    Emax = float(Emax_TeV)
    if Emax <= Emin:
        return 0.0
    if abs(index - 1.0) < 1e-9:
        return K * np.log(Emax / Emin)
    return (K / (1.0 - index)) * (Emax**(1.0 - index) - Emin**(1.0 - index))


# ----------------------- IO and extraction ---------------------------
def load_dl2_data(file_pattern, tel="stereo", zenith_angle=None):
    """
    Load DL2 tables across many files, filter by zenith token, compute theta.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")

    if zenith_angle is not None:
        token = f"_{zenith_angle}_{zenith_angle}deg_"
        files = [f for f in files if token in f]
        if not files:
            raise FileNotFoundError(f"No files found for zenith angle {zenith_angle} degrees")

    tables = []
    for f in files:
        try:
            t = read_table(f, f"/dl2/event/telescope/parameters/{tel}")
            t.meta = {}
            tables.append(t)
        except Exception as e:
            print(f"  Warning: Could not read {Path(f).name}: {e}")
    if not tables:
        raise ValueError("No data could be loaded from DL2 files")

    tab = vstack(tables, metadata_conflicts="silent")

    # Compute theta (rad) from alt/az columns
    theta = _theta_from_altaz(tab["true_alt"], tab["true_az"], tab["reco_alt"], tab["reco_az"])
    tab["theta"] = theta

    # Ensure energies are present as plain arrays for calculations
    if not isinstance(tab["true_energy"], np.ndarray):
        tab["true_energy"] = _to_value_tev(tab["true_energy"])
    if not isinstance(tab["reco_energy"], np.ndarray):
        tab["reco_energy"] = _to_value_tev(tab["reco_energy"])

    return tab


def load_simulation_info(file_pattern):
    """
    Extract minimal throw info across files to build SimulatedEventsInfo.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")

    info = dict(n_showers=0, energy_min=None, energy_max=None, max_impact=None, viewcone=None, spectral_index=-2.0)

    for f in files:
        try:
            sd = read_table(f, "/simulation/service/shower_distribution")
        except Exception as e:
            print(f"  Warning (sim info) {Path(f).name}: {e}")
            continue

        # n_entries per obs_id
        if "n_entries" in sd.colnames:
            info["n_showers"] += int(np.nansum(np.asarray(sd["n_entries"])))

        # energy bins
        if "bins_energy" in sd.colnames and info["energy_min"] is None:
            eb = sd["bins_energy"][0]
            if hasattr(eb, "__len__") and len(eb) > 1:
                info["energy_min"] = float(np.min(eb))  # assume TeV
                info["energy_max"] = float(np.max(eb))

        # core dist bins → max impact (m)
        if "bins_core_dist" in sd.colnames and info["max_impact"] is None:
            cb = sd["bins_core_dist"][0]
            if hasattr(cb, "__len__") and len(cb) > 0:
                info["max_impact"] = float(np.max(cb))

        # viewcone (deg) if present (some sims store this elsewhere; keep optional)
        if "viewcone" in sd.colnames and info["viewcone"] is None:
            try:
                info["viewcone"] = float(sd["viewcone"][0])
            except Exception:
                pass

    # sanity fallbacks
    if info["energy_min"] is None or info["energy_max"] is None:
        info["energy_min"], info["energy_max"] = 0.05, 100.0  # TeV default envelope
    if info["max_impact"] is None:
        info["max_impact"] = 300.0  # meters, conservative default if missing

    return info


def as_sim_info(d):
    """Convert our dict to SimulatedEventsInfo with units."""
    return SimulatedEventsInfo(
        n_showers=int(d["n_showers"]),
        energy_min=float(d["energy_min"]) * u.TeV,
        energy_max=float(d["energy_max"]) * u.TeV,
        max_impact=float(d["max_impact"]) * u.m,
        viewcone=(0.0 if d.get("viewcone") is None else float(d["viewcone"])) * u.deg,
        spectral_index=d.get("spectral_index", -2.0),  # Standard gamma-ray spectral index
    )


# ---------------------- Core computations ---------------------------
def compute_binned_cuts_gamma(gamma_tab, e_true_bins_TeV, gh_eff, th_eff):
    """
    Compute gamma-derived binned cuts (per true-E bin) using pyirf helpers.
    
    For gammaness with operator ">=" we want to keep the upper tail (high gammaness = gamma-like).
    To keep gh_eff fraction of events, we need the (1 - gh_eff) percentile.
    
    For theta with operator "<=" we want to keep the lower tail (small theta = close to source).
    To keep th_eff fraction of events, we use th_eff percentile directly.
    
    Returns:
      gh_cut_table [nbins], th_cut_table [nbins], median scalars (for CSV display)
    """
    e_true = np.asarray(gamma_tab["true_energy"], dtype=float)
    gh = np.asarray(gamma_tab["gammaness"], dtype=float)
    th_deg = np.rad2deg(np.asarray(gamma_tab["theta"], dtype=float))

    # For ">=" operator on gammaness: use (1 - eff) to keep upper tail
    gh_target = 1.0 - float(gh_eff)
    th_target = float(th_eff)
    
    gh_cut = _safe_percentile_cut(gh, e_true, e_true_bins_TeV, gh_target)
    th_cut = _safe_percentile_cut(th_deg, e_true, e_true_bins_TeV, th_target)

    # Fill NaNs (empty bins) by nearest valid value
    def _fill_nearest(x):
        x = np.asarray(x, dtype=float)
        if np.all(~np.isfinite(x)):
            return np.zeros_like(x)
        ii = np.where(np.isfinite(x))[0]
        for j in range(len(x)):
            if not np.isfinite(x[j]):
                jnear = ii[np.argmin(np.abs(ii - j))]
                x[j] = x[jnear]
        return x

    gh_cut = _fill_nearest(gh_cut)
    th_cut = _fill_nearest(th_cut)

    gh_display = float(np.nanmedian(gh_cut))
    th_display = float(np.nanmedian(th_cut))
    return gh_cut, th_cut, gh_display, th_display


def selection_masks(tab, e_true_bins_TeV, gh_cut_table, th_cut_table):
    """Evaluate binned cuts on a table (gamma or proton)."""
    e_true = np.asarray(tab["true_energy"], dtype=float)
    gh = np.asarray(tab["gammaness"], dtype=float)
    th_deg = np.rad2deg(np.asarray(tab["theta"], dtype=float))

    sel_gh = _eval_binned(gh, e_true, e_true_bins_TeV, gh_cut_table, op=">=")
    sel_th = _eval_binned(th_deg, e_true, e_true_bins_TeV, th_cut_table, op="<=")
    return sel_gh & sel_th


def compute_aeff(gamma_tab, sim_gamma, e_true_bins):
    """pyirf.effective_area_per_energy (returns m2 numpy array)."""
    # pyirf API: effective_area_per_energy(selected_events_table, simulation_info, true_energy_bins)
    # selected_events_table must have 'true_energy' column
    aeff = effective_area_per_energy(
        selected_events=gamma_tab,  # Pass the full table
        simulation_info=sim_gamma,
        true_energy_bins=e_true_bins
    )
    return aeff.to_value(u.m**2)


def compute_aeff_with_selection(gamma_tab, sel_mask, sim_gamma, e_true_bins):
    # Pass only the selected rows of the table
    selected_tab = gamma_tab[sel_mask].copy()
    
    # Guarantee Quantity with TeV units (some pyirf versions require this)
    if not hasattr(selected_tab['true_energy'], 'unit'):
        selected_tab['true_energy'] = selected_tab['true_energy'] * u.TeV
    
    aeff = effective_area_per_energy(
        selected_events=selected_tab,
        simulation_info=sim_gamma,
        true_energy_bins=e_true_bins
    )
    return aeff.to_value(u.m**2)


def compute_edisp_params(gamma_tab, sel_mask, e_true_bins_TeV):
    """
    Fit residual distributions per true-energy bin on SELECTED gamma events.
    Returns list of dicts: mu_loc, mu_scale, mu_a, model per bin.
    """
    return _fit_residuals_per_true_bin(
        true_e=np.asarray(gamma_tab["true_energy"], dtype=float),
        reco_e=np.asarray(gamma_tab["reco_energy"], dtype=float),
        selected_mask=np.asarray(sel_mask, dtype=bool),
        e_true_edges=e_true_bins_TeV
    )


def background_rate_shape_only(proton_tab, sel_mask, e_reco_bins_TeV, total_rate_hz=1.0):
    """
    Shape-only background: distribute a user-given total rate across reco-energy bins
    according to selected proton counts.
    
    Adds a minimum floor to prevent division by zero in downstream analysis.
    """
    ereco = np.asarray(proton_tab["reco_energy"], dtype=float)
    counts, _ = np.histogram(ereco[sel_mask], bins=e_reco_bins_TeV)
    counts = counts.astype(float)
    if counts.sum() <= 0:
        return np.full(len(e_reco_bins_TeV) - 1, 1e-12)
    
    rate = total_rate_hz * counts / counts.sum()
    # Add minimum floor to prevent zeros (important for signal/background ratio calculations)
    rate = np.maximum(rate, 1e-12)
    return rate


def background_rate_powerlaw(proton_tab, sel_mask, sim_proton, e_reco_bins_TeV, K, index):
    """
    Approximate absolute background by weighting to a power-law CR flux (TeV units):
      dN/dE [m^-2 s^-1 sr^-1 TeV^-1] = K * E^{-index}

    Rate per bin ≈  ∫_bin dE [K E^{-index}] * (thrown_area [m2]) * (solid_angle [sr]) * (selection_eff(E))
    We estimate selection efficiency(E) from selected/total in reco-energy bins.

    Note: still approximate; without per-event weights and exact thrown distributions
    this is a pragmatic anchor.
    """
    # thrown area and solid angle
    area = np.pi * float(sim_proton.max_impact.to_value(u.m)) ** 2  # m2
    view_deg = float(sim_proton.viewcone.to_value(u.deg))
    omega = _solid_angle_from_viewcone(view_deg) if view_deg > 0 else 2*np.pi  # sr (fallback: 2π)
    ereco = np.asarray(proton_tab["reco_energy"], dtype=float)

    counts_sel, _ = np.histogram(ereco[sel_mask], bins=e_reco_bins_TeV)
    counts_all, _ = np.histogram(ereco, bins=e_reco_bins_TeV)
    eff = np.zeros_like(counts_sel, dtype=float)
    m = counts_all > 0
    eff[m] = counts_sel[m] / counts_all[m]

    rates = np.zeros_like(counts_sel, dtype=float)
    for i in range(len(rates)):
        Emin = e_reco_bins_TeV[i]
        Emax = e_reco_bins_TeV[i+1]
        flux_int = _powerlaw_flux_integral(Emin, Emax, K, index)  # [m^-2 s^-1 sr^-1]
        rates[i] = flux_int * area * omega * eff[i]
    # Ensure strictly positive small floor
    rates[rates <= 0] = 1e-12
    return rates


# ---------------------- CSV Generators -------------------------------
def generate_for_zenith(zenith, gh_eff, th_eff, dl2_dir, tel, out_gamma_path, out_backg_path,
                        bkg_mode="shape", bkg_total_rate=1.0, cr_k=1.8e-7, cr_index=2.7,
                        save_edisp_matrix=False):
    print(f"\n=== Zenith {zenith}°, gh={gh_eff:.2f}, th={th_eff:.2f} ===")

    gamma_pat  = str(Path(dl2_dir) / "gamma_*.h5")
    proton_pat = str(Path(dl2_dir) / "proton_*.h5")

    gamma_tab = load_dl2_data(gamma_pat, tel=tel, zenith_angle=zenith)
    prot_tab  = load_dl2_data(proton_pat, tel=tel, zenith_angle=zenith)

    sim_g = as_sim_info(load_simulation_info(gamma_pat))
    sim_p = as_sim_info(load_simulation_info(proton_pat))

    # Bins (match LST1 resolution: ~20 bins per decade)
    e_true_bins = create_bins_per_decade(0.01 * u.TeV, 40.0 * u.TeV, bins_per_decade=20)
    e_reco_bins = create_bins_per_decade(0.01 * u.TeV, 100.0 * u.TeV, bins_per_decade=20)
    e_true_edges_TeV = e_true_bins.to_value(u.TeV)
    e_reco_edges_TeV = e_reco_bins.to_value(u.TeV)

    # Gamma-derived binned cuts
    gh_cut_table, th_cut_table, gh_disp, th_disp = compute_binned_cuts_gamma(
        gamma_tab, e_true_edges_TeV, gh_eff=gh_eff, th_eff=th_eff
    )

    # Apply to gamma and protons (per true-E bin)
    sel_gamma = selection_masks(gamma_tab, e_true_edges_TeV, gh_cut_table, th_cut_table)
    sel_prot  = selection_masks(prot_tab,  e_true_edges_TeV, gh_cut_table, th_cut_table)

    # Effective area after cuts
    aeff_m2 = compute_aeff_with_selection(gamma_tab, sel_gamma, sim_g, e_true_bins)

    # Energy dispersion parameters per true-E bin (from event residuals on selected gammas)
    emig = compute_edisp_params(gamma_tab, sel_gamma, e_true_edges_TeV)

    # Optional: save edisp matrix (for sanity checks)
    if save_edisp_matrix:
        te = gamma_tab["true_energy"] * (u.TeV if not hasattr(gamma_tab["true_energy"], "unit") else 1)
        re = gamma_tab["reco_energy"] * (u.TeV if not hasattr(gamma_tab["reco_energy"], "unit") else 1)
        ed = energy_dispersion(true_energy=te, reco_energy=re, selected=sel_gamma,
                               true_energy_bins=e_true_bins, reco_energy_bins=e_reco_bins)
        # Save as compressed numpy for inspection
        np.savez_compressed(Path(out_gamma_path).with_suffix(f".zen{zenith}.edisp.npz"),
                            edisp=ed.to_value(u.one), e_true=e_true_edges_TeV, e_reco=e_reco_edges_TeV)

    # ---------- Build Gamma CSV rows ----------
    gamma_rows = []
    for i in range(len(e_true_edges_TeV) - 1):
        # Get emig parameters, use defaults if missing
        mu_loc = emig[i]["mu_loc"] if np.isfinite(emig[i]["mu_loc"]) else 0.0
        mu_scale = emig[i]["mu_scale"] if np.isfinite(emig[i]["mu_scale"]) else 0.2
        mu_a = emig[i]["mu_a"] if np.isfinite(emig[i]["mu_a"]) else ""
        
        row = {
            "ZD_deg": float(zenith),
            "Etrue_min_TeV": float(e_true_edges_TeV[i]),
            "Etrue_max_TeV": float(e_true_edges_TeV[i+1]),
            "Aeff_m2": float(max(aeff_m2[i], 0.0)),
            "emig_mu_loc": mu_loc,
            "emig_mu_scale": mu_scale,
            "emig_mu_a": mu_a,
            "emig_model": emig[i]["model"],
        }
        gamma_rows.append(row)

    # ---------- Background rate ----------
    if bkg_mode == "shape":
        rate = background_rate_shape_only(prot_tab, sel_prot, e_reco_edges_TeV, total_rate_hz=bkg_total_rate)
    else:
        rate = background_rate_powerlaw(prot_tab, sel_prot, sim_p, e_reco_edges_TeV, K=cr_k, index=cr_index)

    # Build Background CSV rows (cuts as single “display” scalars to keep LST format)
    back_rows = []
    for i in range(len(e_reco_edges_TeV) - 1):
        row = {
            "ZD_deg": float(zenith),
            "Ereco_min_TeV": float(e_reco_edges_TeV[i]),
            "Ereco_max_TeV": float(e_reco_edges_TeV[i+1]),
            "BckgRate_per_second": float(rate[i]),
            "Theta_cut_deg": float(th_disp),
            "Gammaness_cut": float(gh_disp),
        }
        back_rows.append(row)

    # Append/Write files
    df_g = pd.DataFrame(gamma_rows)
    df_b = pd.DataFrame(back_rows)

    # Create or append depending on file existence
    if Path(out_gamma_path).exists():
        df_g.to_csv(out_gamma_path, mode="a", header=False, index=False)
    else:
        df_g.to_csv(out_gamma_path, index=False)

    if Path(out_backg_path).exists():
        df_b.to_csv(out_backg_path, mode="a", header=False, index=False)
    else:
        df_b.to_csv(out_backg_path, index=False)


# --------------------------- CLI ------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate approximate IRF CSV files (LST-compatible columns) from SST-1M DL2.")
    # Single values
    p.add_argument("--gh-cut", type=float, default=0.70, help="Gammaness cut efficiency (0..1).")
    p.add_argument("--th-cut", type=float, default=0.70, help="Theta cut efficiency (0..1).")
    p.add_argument("--zenith", type=int, default=None, help="Zenith angle to process (20,30,40,60). If omitted, process all.")
    p.add_argument("--tel", type=str, default="stereo", help="Telescope group for DL2 path: tel_001, tel_002, stereo.")

    # Grid
    p.add_argument("--grid", type=str, default=None,
                   help="Comma-separated efficiencies to generate for both cuts (e.g., '0.4,0.7,0.9'). Overrides --gh-cut/--th-cut.")

    # Paths
    p.add_argument("--dl2-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\dl2_gamma",
                   help="Folder containing gamma_*.h5 and proton_*.h5")
    p.add_argument("--output-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\SST1M_csv",
                   help="Output folder for CSVs")
    p.add_argument("--lst-ref-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\lst_data",
                   help="(Optional) LST reference CSVs (not used by the generator).")

    # Background options
    p.add_argument("--bkg-mode", choices=["shape", "powerlaw"], default="shape",
                   help="Background method: 'shape' uses MC shape + --bkg-total-rate; 'powerlaw' uses CR weighting.")
    p.add_argument("--bkg-total-rate", type=float, default=1.0,
                   help="Total background rate within ROI [Hz] for 'shape' mode (distributed across reco bins).")

    p.add_argument("--cr-powerlaw", action="store_true",
                   help="Shortcut to set --bkg-mode powerlaw (same as --bkg-mode powerlaw).")
    p.add_argument("--cr-k", type=float, default=1.8e-7,
                   help="Power-law normalization K in TeV units: dN/dE = K * E^-index [m^-2 s^-1 sr^-1 TeV^-1].")
    p.add_argument("--cr-index", type=float, default=2.7,
                   help="Power-law spectral index for CR proton flux.")

    # Debug
    p.add_argument("--save-edisp-matrix", action="store_true",
                   help="Save edisp matrices (npz) per zenith for inspection.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.grid:
        effs = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
        gh_list = effs
        th_list = effs
    else:
        gh_list = [args.gh_cut]
        th_list = [args.th_cut]

    if args.cr_powerlaw:
        args.bkg_mode = "powerlaw"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    zeniths = [args.zenith] if args.zenith else [20, 30, 40, 60]

    print("=" * 84)
    print("SST-1M Approximate IRF (LST CSV format)")
    print("=" * 84)
    print(f"DL2 dir       : {args.dl2_dir}")
    print(f"Output dir    : {outdir}")
    print(f"Telescope node: {args.tel}")
    print(f"Zeniths       : {zeniths}")
    print(f"Mode (bkg)    : {args.bkg_mode}")
    if args.bkg_mode == "shape":
        print(f"  total rate  : {args.bkg_total_rate} Hz")
    else:
        print(f"  CR K/index  : {args.cr_k} , {args.cr_index}")
    print("=" * 84)

    for gh in gh_list:
        for th in th_list:
            # output filenames per efficiency pair
            gamma_csv = outdir / f"SST1M_gamma_irf_gheffi_{gh:.2f}_theffi_{th:.2f}.csv"
            backg_csv = outdir / f"SST1M_backg_irf_gheffi_{gh:.2f}_theffi_{th:.2f}.csv"

            # Remove existing to avoid appending leftovers when mixing zeniths across runs
            if gamma_csv.exists():
                gamma_csv.unlink()
            if backg_csv.exists():
                backg_csv.unlink()

            for zen in zeniths:
                try:
                    generate_for_zenith(
                        zenith=zen, gh_eff=gh, th_eff=th,
                        dl2_dir=args.dl2_dir, tel=args.tel,
                        out_gamma_path=str(gamma_csv),
                        out_backg_path=str(backg_csv),
                        bkg_mode=args.bkg_mode,
                        bkg_total_rate=args.bkg_total_rate,
                        cr_k=args.cr_k, cr_index=args.cr_index,
                        save_edisp_matrix=args.save_edisp_matrix
                    )
                except Exception as e:
                    print(f"  [WARN] Zenith {zen}: {e}")

            print(f"Saved gamma CSV : {gamma_csv}")
            print(f"Saved backg CSV : {backg_csv}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
