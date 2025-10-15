#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import glob
import inspect

import astropy.units as u
from astropy.table import Table, vstack

from scipy.stats import moyal, skewnorm

if sys.platform == 'win32':
    import types
    sys.modules['pwd'] = types.ModuleType('pwd')
    sys.modules['pwd'].getpwuid = lambda x: types.SimpleNamespace(pw_name='user')

from ctapipe.io import read_table

from pyirf.irf import energy_dispersion, effective_area_per_energy
from pyirf.simulations import SimulatedEventsInfo
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.binning import create_bins_per_decade


def _to_value_tev(arr):
    if hasattr(arr, 'unit'):
        if hasattr(arr, 'quantity'):
            return np.asarray(arr.quantity.to_value(u.TeV))
        return np.asarray(arr.to_value(u.TeV))
    return np.asarray(arr, dtype=float)

def _to_value_rad(arr):
    if hasattr(arr, 'unit'):
        if hasattr(arr, 'quantity'):
            qty = arr.quantity
            if qty.unit == u.dimensionless_unscaled or str(qty.unit) == '':
                return np.asarray(qty.value)
            return np.asarray(qty.to_value(u.rad))
        else:
            if arr.unit == u.dimensionless_unscaled or str(arr.unit) == '':
                return np.asarray(arr.value)
            return np.asarray(arr.to_value(u.rad))
    return np.asarray(arr, dtype=float)

def _theta_from_altaz(true_alt, true_az, reco_alt, reco_az):
    talt = _to_value_rad(true_alt)
    taz  = _to_value_rad(true_az)
    ralt = _to_value_rad(reco_alt)
    raz  = _to_value_rad(reco_az)
    cos_theta = (np.sin(talt) * np.sin(ralt)
                 + np.cos(talt) * np.cos(ralt) * np.cos(taz - raz))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def _safe_percentile_cut(values, bin_values, bins, eff):
    v = np.asarray(values, dtype=float)
    bv = np.asarray(bin_values, dtype=float)
    edges = np.asarray(bins, dtype=float)
    try:
        sig = inspect.signature(calculate_percentile_cut)
        if 'efficiency' in sig.parameters:
            return calculate_percentile_cut(values=v, bins=edges, bin_values=bv, efficiency=float(eff))
        elif 'percentile' in sig.parameters:
            return calculate_percentile_cut(values=v, bins=edges, bin_values=bv, percentile=float(eff) * 100.0)
    except Exception:
        pass
    cut = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (bv >= edges[i]) & (bv < edges[i+1])
        if np.any(m):
            cut[i] = np.percentile(v[m], eff * 100.0)
    return cut

def _eval_binned(values, bin_values, bins, cut, op):
    v  = np.asarray(values, dtype=float)
    bv = np.asarray(bin_values, dtype=float)
    edges = np.asarray(bins, dtype=float)
    try:
        return evaluate_binned_cut(values=v, bins=edges, bin_values=bv, cut=cut, operator=op)
    except Exception:
        sel = np.zeros_like(v, dtype=bool)
        idx = np.digitize(bv, edges) - 1
        good = (idx >= 0) & (idx < len(cut))
        if op == ">=":
            sel[good] = v[good] >= cut[idx[good]]
        else:
            sel[good] = v[good] <= cut[idx[good]]
        return sel

def _fit_residuals_per_true_bin(true_e, reco_e, selected_mask, e_true_edges):
    te  = np.asarray(true_e, dtype=float)
    re  = np.asarray(reco_e, dtype=float)
    sel = np.asarray(selected_mask, dtype=bool)
    n_bins = len(e_true_edges) - 1
    out = []
    for i in range(n_bins):
        m = sel & (te >= e_true_edges[i]) & (te < e_true_edges[i+1])
        if m.sum() < 20:
            out.append(dict(mu_loc=np.nan, mu_scale=np.nan, mu_a=np.nan, model="log10_moyal"))
            continue
        rel = np.log10(np.clip(re[m], 1e-20, None) / np.clip(te[m], 1e-20, None))
        med = np.median(rel)
        mad = np.median(np.abs(rel - med))
        sigma = max(1.4826 * mad, 1e-3)
        clean = rel[np.abs(rel - med) < 3.0 * sigma]
        if clean.size < 20:
            clean = rel
        try:
            loc_m, scale_m = moyal.fit(clean, loc=med, scale=sigma)
        except Exception:
            loc_m, scale_m = med, sigma
        ok_m = np.isfinite(loc_m) and np.isfinite(scale_m) and (abs(loc_m) < 2.0) and (0.005 < scale_m < 3.0)
        try:
            a_s, loc_s, scale_s = skewnorm.fit(clean, loc=med, scale=sigma)
            ok_s = (np.isfinite(a_s) and np.isfinite(loc_s) and np.isfinite(scale_s)
                    and abs(a_s) < 30 and abs(loc_s) < 2.0 and 0.005 < scale_s < 3.0)
        except Exception:
            ok_s = False
            a_s = loc_s = scale_s = np.nan
        if ok_s and (abs(a_s) > 0.5):
            out.append(dict(mu_loc=float(loc_s), mu_scale=float(scale_s), mu_a=float(a_s), model="log10_skewnorm"))
        elif ok_m:
            out.append(dict(mu_loc=float(loc_m), mu_scale=float(scale_m), mu_a=np.nan, model="log10_moyal"))
        else:
            out.append(dict(mu_loc=float(med),  mu_scale=float(sigma),  mu_a=np.nan, model="log10_moyal"))
    return out

def _solid_angle_from_viewcone(viewcone_deg):
    v = float(max(viewcone_deg, 0.0))
    return 2.0 * np.pi * (1.0 - np.cos(np.deg2rad(v)))

def _powerlaw_flux_integral(Emin_TeV, Emax_TeV, K, index):
    Emin = float(Emin_TeV)
    Emax = float(Emax_TeV)
    if Emax <= Emin:
        return 0.0
    if abs(index - 1.0) < 1e-9:
        return K * np.log(Emax / Emin)
    return (K / (1.0 - index)) * (Emax**(1.0 - index) - Emin**(1.0 - index))

def load_dl2_data(file_pattern, tel="stereo", zenith_angle=None):
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
    theta = _theta_from_altaz(tab["true_alt"], tab["true_az"], tab["reco_alt"], tab["reco_az"])
    tab["theta"] = theta
    if not isinstance(tab["true_energy"], np.ndarray):
        tab["true_energy"] = _to_value_tev(tab["true_energy"])
    if not isinstance(tab["reco_energy"], np.ndarray):
        tab["reco_energy"] = _to_value_tev(tab["reco_energy"])
    return tab

def load_simulation_info(file_pattern):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")
    info = dict(n_showers=0, energy_min=None, energy_max=None, max_impact=None, viewcone=None, spectral_index=-2.0)
    for f in files:
        try:
            sd = read_table(f, "/simulation/service/shower_distribution")
        except Exception:
            sd = None
        if sd is not None:
            if "n_entries" in sd.colnames:
                info["n_showers"] += int(np.nansum(np.asarray(sd["n_entries"])))
            if "bins_energy" in sd.colnames and info["energy_min"] is None:
                eb = sd["bins_energy"][0]
                if hasattr(eb, "__len__") and len(eb) > 1:
                    info["energy_min"] = float(np.min(eb))
                    info["energy_max"] = float(np.max(eb))
            if "bins_core_dist" in sd.colnames and info["max_impact"] is None:
                cb = sd["bins_core_dist"][0]
                if hasattr(cb, "__len__") and len(cb) > 0:
                    info["max_impact"] = float(np.max(cb))
            if "viewcone" in sd.colnames and info["viewcone"] is None:
                try:
                    info["viewcone"] = float(sd["viewcone"][0])
                except Exception:
                    pass
    if info["energy_min"] is None or info["energy_max"] is None:
        info["energy_min"], info["energy_max"] = 0.05, 100.0
    if info["max_impact"] is None:
        info["max_impact"] = 300.0
    return info

def as_sim_info(d):
    return SimulatedEventsInfo(
        n_showers=int(d["n_showers"]),
        energy_min=float(d["energy_min"]) * u.TeV,
        energy_max=float(d["energy_max"]) * u.TeV,
        max_impact=float(d["max_impact"]) * u.m,
        viewcone=(0.0 if d.get("viewcone") is None else float(d["viewcone"])) * u.deg,
        spectral_index=d.get("spectral_index", -2.0),
    )

def compute_binned_cuts_gamma(gamma_tab, e_true_bins_TeV, gh_eff, th_eff):
    e_true = np.asarray(gamma_tab["true_energy"], dtype=float)
    gh = np.asarray(gamma_tab["gammaness"], dtype=float)
    th_deg = np.rad2deg(np.asarray(gamma_tab["theta"], dtype=float))
    gh_target = 1.0 - float(gh_eff)
    th_target = float(th_eff)
    gh_cut = _safe_percentile_cut(gh, e_true, e_true_bins_TeV, gh_target)
    th_cut = _safe_percentile_cut(th_deg, e_true, e_true_bins_TeV, th_target)
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
    e_true = np.asarray(tab["true_energy"], dtype=float)
    gh = np.asarray(tab["gammaness"], dtype=float)
    th_deg = np.rad2deg(np.asarray(tab["theta"], dtype=float))
    sel_gh = _eval_binned(gh, e_true, e_true_bins_TeV, gh_cut_table, op=">=")
    sel_th = _eval_binned(th_deg, e_true, e_true_bins_TeV, th_cut_table, op="<=")
    return sel_gh & sel_th

def compute_aeff_with_selection(gamma_tab, sel_mask, sim_gamma, e_true_bins):
    selected_tab = gamma_tab[sel_mask].copy()
    if not hasattr(selected_tab['true_energy'], 'unit'):
        selected_tab['true_energy'] = selected_tab['true_energy'] * u.TeV
    aeff = effective_area_per_energy(
        selected_events=selected_tab,
        simulation_info=sim_gamma,
        true_energy_bins=e_true_bins
    )
    return aeff.to_value(u.m**2)

def compute_edisp_params(gamma_tab, sel_mask, e_true_bins_TeV):
    return _fit_residuals_per_true_bin(
        true_e=np.asarray(gamma_tab["true_energy"], dtype=float),
        reco_e=np.asarray(gamma_tab["reco_energy"], dtype=float),
        selected_mask=np.asarray(sel_mask, dtype=bool),
        e_true_edges=e_true_bins_TeV
    )

def background_rate_shape_only(proton_tab, sel_mask, e_reco_bins_TeV, total_rate_hz=1.0):
    ereco = np.asarray(proton_tab["reco_energy"], dtype=float)
    counts, _ = np.histogram(ereco[sel_mask], bins=e_reco_bins_TeV)
    counts = counts.astype(float)
    if counts.sum() <= 0:
        return np.full(len(e_reco_bins_TeV) - 1, 1e-12)
    rate = total_rate_hz * counts / counts.sum()
    return np.maximum(rate, 1e-12)

def background_rate_powerlaw(proton_tab, sel_mask, sim_proton, e_reco_bins_TeV, K, index):
    area = np.pi * float(sim_proton.max_impact.to_value(u.m)) ** 2
    view_deg = float(sim_proton.viewcone.to_value(u.deg))
    omega = _solid_angle_from_viewcone(view_deg) if view_deg > 0 else 2*np.pi
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
        flux_int = _powerlaw_flux_integral(Emin, Emax, K, index)
        rates[i] = flux_int * area * omega * eff[i]
    rates[rates <= 0] = 1e-12
    return rates

def generate_for_zenith(zenith, gh_eff, th_eff, dl2_dir, tel, out_gamma_path, out_backg_path,
                        bkg_mode="shape", bkg_total_rate=1.0, cr_k=1.8e-7, cr_index=2.7,
                        save_edisp_matrix=False):
    gamma_pat  = str(Path(dl2_dir) / "gamma_*.h5")
    proton_pat = str(Path(dl2_dir) / "proton_*.h5")
    gamma_tab = load_dl2_data(gamma_pat, tel=tel, zenith_angle=zenith)
    prot_tab  = load_dl2_data(proton_pat, tel=tel, zenith_angle=zenith)
    sim_g = as_sim_info(load_simulation_info(gamma_pat))
    sim_p = as_sim_info(load_simulation_info(proton_pat))
    assert sim_g.n_showers > 0, "Gamma n_showers=0; check simulation metadata"
    assert sim_p.n_showers > 0, "Proton n_showers=0; check simulation metadata"
    e_true_bins = create_bins_per_decade(0.01 * u.TeV, 40.0 * u.TeV, bins_per_decade=20)
    e_reco_bins = create_bins_per_decade(0.01 * u.TeV, 100.0 * u.TeV, bins_per_decade=20)
    e_true_edges_TeV = e_true_bins.to_value(u.TeV)
    e_reco_edges_TeV = e_reco_bins.to_value(u.TeV)
    gh_cut_table, th_cut_table, gh_disp, th_disp = compute_binned_cuts_gamma(
        gamma_tab, e_true_edges_TeV, gh_eff=gh_eff, th_eff=th_eff
    )
    sel_gamma = selection_masks(gamma_tab, e_true_edges_TeV, gh_cut_table, th_cut_table)
    sel_prot  = selection_masks(prot_tab,  e_true_edges_TeV, gh_cut_table, th_cut_table)
    aeff_m2 = compute_aeff_with_selection(gamma_tab, sel_gamma, sim_g, e_true_bins)
    emig = compute_edisp_params(gamma_tab, sel_gamma, e_true_edges_TeV)
    if save_edisp_matrix:
        te = gamma_tab["true_energy"] * (u.TeV if not hasattr(gamma_tab["true_energy"], "unit") else 1)
        re = gamma_tab["reco_energy"] * (u.TeV if not hasattr(gamma_tab["reco_energy"], "unit") else 1)
        ed = energy_dispersion(true_energy=te, reco_energy=re, selected=sel_gamma,
                               true_energy_bins=e_true_bins, reco_energy_bins=e_reco_bins)
        np.savez_compressed(Path(out_gamma_path).with_suffix(f".zen{zenith}.edisp.npz"),
                            edisp=ed.to_value(u.one), e_true=e_true_edges_TeV, e_reco=e_reco_edges_TeV)
    gamma_rows = []
    for i in range(len(e_true_edges_TeV) - 1):
        mu_loc = emig[i]["mu_loc"] if np.isfinite(emig[i]["mu_loc"]) else 0.0
        mu_scale = emig[i]["mu_scale"] if np.isfinite(emig[i]["mu_scale"]) else 0.2
        mu_a = float(emig[i]["mu_a"]) if np.isfinite(emig[i]["mu_a"]) else np.nan
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
    if bkg_mode == "shape":
        rate = background_rate_shape_only(prot_tab, sel_prot, e_reco_edges_TeV, total_rate_hz=bkg_total_rate)
    else:
        rate = background_rate_powerlaw(prot_tab, sel_prot, sim_p, e_reco_edges_TeV, K=cr_k, index=cr_index)
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
    df_g = pd.DataFrame(gamma_rows)
    df_b = pd.DataFrame(back_rows)
    if Path(out_gamma_path).exists():
        df_g.to_csv(out_gamma_path, mode="a", header=False, index=False)
    else:
        df_g.to_csv(out_gamma_path, index=False)
    if Path(out_backg_path).exists():
        df_b.to_csv(out_backg_path, mode="a", header=False, index=False)
    else:
        df_b.to_csv(out_backg_path, index=False)

def parse_args():
    p = argparse.ArgumentParser(description="Generate SST-1M IRF CSVs (LST-compatible).")
    p.add_argument("--gh-cut", type=float, default=0.70)
    p.add_argument("--th-cut", type=float, default=0.70)
    p.add_argument("--zenith", type=int, default=None)
    p.add_argument("--tel", type=str, default="stereo")
    p.add_argument("--grid", type=str, default=None)
    p.add_argument("--dl2-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\dl2_gamma")
    p.add_argument("--output-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\SST1M_csv")
    p.add_argument("--lst-ref-dir", type=str,
                   default=r"C:\Users\a.nakhle\sst1mpipe\sst1mpipe\source_simulation\lst_data")
    p.add_argument("--bkg-mode", choices=["shape", "powerlaw"], default="shape")
    p.add_argument("--bkg-total-rate", type=float, default=1.0)
    p.add_argument("--cr-powerlaw", action="store_true")
    p.add_argument("--cr-k", type=float, default=1.8e-7)
    p.add_argument("--cr-index", type=float, default=2.7)
    p.add_argument("--save-edisp-matrix", action="store_true")
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
            gamma_csv = outdir / f"SST1M_gamma_irf_gheffi_{gh:.2f}_theffi_{th:.2f}.csv"
            backg_csv = outdir / f"SST1M_backg_irf_gheffi_{gh:.2f}_theffi_{th:.2f}.csv"
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
