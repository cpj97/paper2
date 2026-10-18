"""Time-varying pre-period covariates with the same window for treated and control cells.

Notebook 04 freezes time-varying covariates (DMSP nightlights, coca) over each treated cell's own pre-treatment
years (years < first_treat_year) but over a fixed window for never-treated cells (years <= 2013). In a
Callaway-Sant'Anna comparison of cohort g with never-treated cells, the two groups' covariates then cover
different calendar years, so part of their difference is the window, not the cell (nightlights grow over time).

Here every cell gets, for every panel year t, the mean of the covariate over the source years <= t (the
running pre-mean). R `did` takes covariates from each comparison's base period (g-1 for post-treatment ATT(g,t),
g-1-anticipation with anticipation, t-1 for the pre-periods with the default varying base), checked on synthetic
data, so treated and control cells of every comparison are measured over the same years. For treated cells at
year g-1 the running mean equals notebook 04's pre-treatment mean exactly (checked by `check_against_frozen`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cumulative_means(values: np.ndarray, src_years: np.ndarray, years: np.ndarray) -> np.ndarray:
    """(n_cells x len(years)): at year t, the mean of `values` (n_cells x len(src_years)) over src_years <= t,
    ignoring missing values; NaN where no source year <= t has data."""
    order = np.argsort(src_years)
    v, sy = values[:, order], np.asarray(src_years)[order]
    ok = np.isfinite(v)
    csum = np.cumsum(np.where(ok, v, 0.0), axis=1)
    ccnt = np.cumsum(ok, axis=1)
    out = np.full((v.shape[0], len(years)), np.nan)
    for j, t in enumerate(years):
        k = np.searchsorted(sy, t, side="right") - 1      # last source year <= t
        if k >= 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, j] = np.where(ccnt[:, k] > 0, csum[:, k] / np.maximum(ccnt[:, k], 1), np.nan)
    return out


def dmsp_annual(cell_ids: np.ndarray, gee_dir: Path, prefix: str = "core_covariates_1km_ntl") -> tuple[np.ndarray, np.ndarray]:
    """Annual DMSP stable lights from notebook 04's Earth Engine exports, aligned to cell_ids (retry chunks preferred)."""
    files = {}
    for p in sorted(Path(gee_dir).glob(f"{prefix}*.csv")):
        key = p.name.replace("_retry.csv", ".csv")
        if key not in files or "_retry" in p.name:
            files[key] = p
    df = pd.concat([pd.read_csv(p) for p in files.values()], ignore_index=True).drop_duplicates("cell_id")
    cols = sorted((int(c.rsplit("_", 1)[1]), c) for c in df.columns if c.startswith("dmsp_stable_lights_") and c.rsplit("_", 1)[1].isdigit())
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.set_index("cell_id").reindex(cell_ids)
    return df[[c for _, c in cols]].to_numpy(dtype=float), np.array([y for y, _ in cols])


def coca_annual(cell_ids: np.ndarray, coca_panel_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annual coca hectares (notebook 04's panel_coca), aligned to cell_ids; also a flag for cells outside the
    monitored Grillas extent (no coca data in any year), which notebook 04 leaves missing rather than zero."""
    c = pd.read_parquet(coca_panel_path, columns=["cell_id", "year", "coca_ha"])
    c["cell_id"] = c["cell_id"].astype(str)
    wide = c.pivot_table(index="cell_id", columns="year", values="coca_ha", aggfunc="sum")
    src_years = wide.columns.to_numpy(dtype=int)
    wide = wide.reindex(cell_ids)
    missing = wide.isna().all(axis=1).to_numpy()
    return wide.to_numpy(dtype=float), src_years, missing


def check_against_frozen(running: np.ndarray, years: np.ndarray, frozen: np.ndarray, g_raw: np.ndarray,
                         never_cutoff: int = 2013, tol: float = 1e-6) -> dict:
    """Compare the running mean with notebook 04's frozen covariate: at year g-1 for cells treated in the panel,
    at `never_cutoff` for never-treated cells. Returns max absolute differences and the cells compared."""
    out = {}
    tr = (g_raw > years.min()) & np.isfinite(frozen)
    j = np.searchsorted(years, g_raw[tr] - 1)
    d_tr = np.abs(running[np.flatnonzero(tr), j] - frozen[tr])
    nv = (g_raw == 0) & np.isfinite(frozen)
    d_nv = np.abs(running[nv, int(np.flatnonzero(years == never_cutoff)[0])] - frozen[nv])
    out["treated_max_abs_diff"] = float(np.nanmax(d_tr)) if d_tr.size else np.nan
    out["treated_share_match"] = float(np.mean(d_tr <= tol)) if d_tr.size else np.nan
    out["never_max_abs_diff"] = float(np.nanmax(d_nv)) if d_nv.size else np.nan
    out["never_share_match"] = float(np.mean(d_nv <= tol)) if d_nv.size else np.nan
    return out


def build_dr_covariates(cell_ids: np.ndarray, years: np.ndarray, g_raw: np.ndarray, *, covariate_path: Path, gee_dir: Path,
                        coca_panel_path: Path, geography: list, nightlights: str = "dmsp_stable_lights_pre_mean",
                        coca: str = "coca_pre_mean_ha", coca_flag: str = "coca_data_missing", verbose: bool = True,
                        chen_path: Path | None = None) -> dict:
    """The DR covariates of notebooks 09, 12 and 13, aligned to cell_ids.

    nightlights: "dmsp_stable_lights_pre_mean" (DMSP 2001-2013, notebook 04's Earth Engine exports) or "chen_ntl_pre_mean"
    (harmonised NPP-VIIRS-like series 2000-2025 of Chen et al. 2021, notebook 04c; needs chen_path).

    Returns {"static": DataFrame (geography + coca flag; R cells.csv), "by_year": {name: cells x years} (nightlights and coca
    running pre-means; R long.csv), "names": all covariates in model order, "x_at": function(base_year) -> (X, complete)
    for the Python estimator, "frozen": DataFrame of notebook 04's frozen covariates (DMSP, coca 0 + flag outside the
    monitored extent), "frozen_names": its columns, "dmsp_by_year": the DMSP running pre-mean (for robustness runs)}.
    Asserts that the DMSP and coca running means equal notebook 04's values at g-1.
    """
    dmsp_name = "dmsp_stable_lights_pre_mean"
    cov = pd.read_parquet(covariate_path)
    cov["cell_id"] = cov["cell_id"].astype(str)
    cov = cov.set_index("cell_id").reindex(cell_ids)
    dm, dy = dmsp_annual(cell_ids, gee_dir)
    cc, cy, outside = coca_annual(cell_ids, coca_panel_path)
    dmsp_tv = cumulative_means(dm, dy, years)
    tv = {coca: cumulative_means(cc, cy, years)}
    for name, m in ((dmsp_name, dmsp_tv), (coca, tv[coca])):
        chk = check_against_frozen(m, years, cov[name].to_numpy(dtype=float), g_raw)
        if verbose:
            print(f"{name}: running mean at g-1 = notebook 04 value for {chk['treated_share_match']:.1%} of treated cells "
                  f"(max diff {chk['treated_max_abs_diff']:.1e}); never-treated at 2013: {chk['never_share_match']:.1%}")
        assert chk["treated_share_match"] == 1.0, f"running mean of {name} does not reproduce notebook 04"
    if nightlights == dmsp_name:
        tv = {nightlights: dmsp_tv, **tv}
    elif nightlights.startswith("chen"):
        ch = pd.read_parquet(chen_path)
        ch["cell_id"] = ch["cell_id"].astype(str)
        cols = sorted((int(c.rsplit("_", 1)[1]), c) for c in ch.columns if c.startswith("chen_ntl_") and c.rsplit("_", 1)[1].isdigit())
        vals = ch.set_index("cell_id").reindex(cell_ids)[[c for _, c in cols]].to_numpy(dtype=float)
        tv = {nightlights: cumulative_means(vals, np.array([y for y, _ in cols]), years), **tv}
        if verbose:
            print(f"{nightlights}: Chen et al. running pre-mean from {cols[0][0]}; cells without data {int(np.isnan(vals).all(axis=1).sum()):,}")
    else:
        raise ValueError(f"unknown nightlights covariate {nightlights}")
    tv[coca] = np.where(outside[:, None], 0.0, np.nan_to_num(tv[coca], nan=0.0))
    static = cov[list(geography)].copy()
    static[coca_flag] = outside.astype(int)
    static = static.reset_index(drop=True)
    names = list(geography) + [nightlights, coca, coca_flag]
    frozen_names = list(geography) + [dmsp_name, coca, coca_flag]
    frozen = cov[list(geography) + [dmsp_name, coca]].copy()
    frozen[coca_flag] = frozen[coca].isna().astype(int)
    frozen[coca] = frozen[coca].fillna(0.0)
    frozen = frozen[frozen_names].reset_index(drop=True)

    def x_at(base_year: int):
        j = int(np.searchsorted(years, base_year))
        cols = {**{c: static[c].to_numpy(dtype=float) for c in static.columns}, **{c: m[:, j] for c, m in tv.items()}}
        X = np.column_stack([cols[c] for c in names])
        return X, ~np.isnan(X).any(axis=1)

    return {"static": static, "by_year": tv, "names": names, "x_at": x_at, "frozen": frozen, "frozen_names": frozen_names,
            "dmsp_by_year": dmsp_tv}
