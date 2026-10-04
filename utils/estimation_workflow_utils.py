from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def append_tag_to_filename(filename: str, tag: str | None) -> str:
    if not tag:
        return filename
    path = Path(filename)
    return f"{path.stem}_{tag}{path.suffix}"


def infer_run_tag(path_like: str | Path) -> str:
    stem = Path(path_like).stem
    for tag in ("1km", "0p080"):
        if stem.endswith(f"_{tag}") or f"_{tag}_" in stem:
            return tag
    return ""


def resolve_path(root: Path, configured: str | Path | None, fallback: str) -> Path:
    if configured is not None:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    matches = sorted((root / "data" / "intermediate").glob(fallback), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match {fallback!r} in data/intermediate.")
    return matches[0]


def treatment_group_for_did(series: pd.Series) -> pd.Series:
    group = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return group.where(group > 0, 0)


def assign_distance_ring(distance_km: pd.Series, rings_km: list[tuple[float, float]]) -> pd.Series:
    out = pd.Series(pd.NA, index=distance_km.index, dtype="string")
    for inner, outer in rings_km:
        label = f"{inner:g}_{outer:g}km"
        out = out.mask((distance_km > float(inner)) & (distance_km <= float(outer)), label)
    out = out.mask(distance_km.isna(), "no_treated_in_year")
    return out


def compute_nearest_treated_exposure(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    treated_col: str,
    rings_km: list[tuple[float, float]],
    buffer_radii_km: Iterable[float],
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    centroid_cols = [cell_col, "centroid_x_m", "centroid_y_m"]
    coords_df = centroids[centroid_cols].drop_duplicates(cell_col).copy()
    coords_df[cell_col] = coords_df[cell_col].astype("string")
    coords = coords_df[["centroid_x_m", "centroid_y_m"]].to_numpy(dtype=float)
    cell_ids = coords_df[cell_col].to_numpy()
    cell_pos = pd.Series(np.arange(len(coords_df)), index=coords_df[cell_col])

    status = panel[[cell_col, year_col, treated_col]].copy()
    status[cell_col] = status[cell_col].astype("string")
    status[year_col] = pd.to_numeric(status[year_col], errors="coerce").astype(int)
    status[treated_col] = pd.to_numeric(status[treated_col], errors="coerce").fillna(0).astype(int)
    if years is None:
        years = sorted(status[year_col].unique())
    else:
        years = sorted(int(y) for y in years)

    out = []
    for year in years:
        year_status = status.loc[status[year_col] == year, [cell_col, treated_col]]
        treated_ids = year_status.loc[year_status[treated_col] == 1, cell_col]
        treated_idx = cell_pos.reindex(treated_ids).dropna().astype(int).to_numpy()
        if len(treated_idx) == 0:
            nearest_km = np.full(len(coords_df), np.nan)
        else:
            tree = cKDTree(coords[treated_idx])
            nearest_m, _ = tree.query(coords, k=1)
            nearest_km = nearest_m / 1000.0

        year_out = pd.DataFrame(
            {
                cell_col: cell_ids,
                year_col: year,
                "nearest_treated_distance_km": nearest_km,
            }
        )
        year_out["distance_ring"] = assign_distance_ring(year_out["nearest_treated_distance_km"], rings_km)
        for radius in buffer_radii_km:
            col = f"outside_{int(radius)}km_buffer"
            year_out[col] = year_out["nearest_treated_distance_km"].isna() | (
                year_out["nearest_treated_distance_km"] > float(radius)
            )
        out.append(year_out)
    return pd.concat(out, ignore_index=True)


def build_stable_buffered_did_panel(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    buffer_km: float,
    cell_col: str,
    year_col: str,
    first_treat_col: str,
    treated_before_panel_col: str,
    outcome_col: str,
    min_year: int,
    max_year: int | None = None,
) -> pd.DataFrame:
    use_cols = [cell_col, year_col, outcome_col, first_treat_col, treated_before_panel_col, "never_treated", "ever_treated"]
    keep = panel[use_cols].copy()
    keep[cell_col] = keep[cell_col].astype("string")
    keep[year_col] = pd.to_numeric(keep[year_col], errors="coerce").astype(int)
    keep = keep[keep[year_col] >= int(min_year)].copy()
    if max_year is not None:
        keep = keep[keep[year_col] <= int(max_year)].copy()

    exp = exposure[[cell_col, year_col, "nearest_treated_distance_km"]].copy()
    exp[cell_col] = exp[cell_col].astype("string")
    exp[year_col] = pd.to_numeric(exp[year_col], errors="coerce").astype(int)
    keep = keep.merge(exp, on=[cell_col, year_col], how="left")

    min_control_distance = (
        keep.loc[keep["never_treated"].astype(bool)]
        .groupby(cell_col)["nearest_treated_distance_km"]
        .min()
        .rename("min_nearest_treated_distance_km")
    )
    clean_controls = min_control_distance[min_control_distance > float(buffer_km)].index

    keep["first_treat_for_did"] = treatment_group_for_did(keep[first_treat_col])
    pre_panel = keep[treated_before_panel_col].fillna(False).astype(bool)
    in_window_treated = (keep["first_treat_for_did"] > int(min_year)) & ~pre_panel
    clean_never_control = keep[cell_col].isin(clean_controls)
    did = keep.loc[in_window_treated | clean_never_control].copy()
    did.loc[clean_never_control, "first_treat_for_did"] = 0
    return did.drop(columns=["nearest_treated_distance_km"])


def build_boundary_band_did_panel(
    panel: pd.DataFrame,
    border_distance: pd.DataFrame,
    *,
    min_km: float,
    max_km: float,
    cell_col: str,
    year_col: str,
    first_treat_col: str,
    treated_before_panel_col: str,
    outcome_col: str,
    min_year: int,
    max_year: int | None = None,
) -> pd.DataFrame:
    """Buffered DiD panel with a MIN/MAX control-distance band instead of a one-sided moat.

    Treatment side is identical to `build_stable_buffered_did_panel`: every cell whose
    first_treat_for_did falls inside the analysis window is kept as treated, regardless of
    how close to or far from its own PA's border it sits -- the band only ever filters the
    control side. Controls are never-treated cells whose distance to the nearest ACTIVE PA
    border (`border_distance`, notebook 07's nearest_active_pa_border_km table, already
    limited to a bounded candidate-search radius) falls in [min_km, max_km] at its closest
    approach over the analysis window.

    nearest_active_pa_border_km only decreases over time (notebook 07 builds it as a running
    minimum over an ever-growing set of active cohorts), so a cell's minimum across the
    window IS its closest-ever approach -- the same statistic build_stable_buffered_did_panel's
    one-sided moat already tests (`min_control_distance > buffer_km`), just capped above as
    well as below. A year where the cell isn't in `border_distance` at all means no PA was
    within the candidate-search radius yet, not that the cell was "too far" -- pandas' default
    skipna drops those years from the min rather than forcing a false violation.
    """
    use_cols = [cell_col, year_col, outcome_col, first_treat_col, treated_before_panel_col, "never_treated", "ever_treated"]
    keep = panel[use_cols].copy()
    keep[cell_col] = keep[cell_col].astype("string")
    keep[year_col] = pd.to_numeric(keep[year_col], errors="coerce").astype(int)
    keep = keep[keep[year_col] >= int(min_year)].copy()
    if max_year is not None:
        keep = keep[keep[year_col] <= int(max_year)].copy()

    bd = border_distance[[cell_col, year_col, "nearest_active_pa_border_km"]].copy()
    bd[cell_col] = bd[cell_col].astype("string")
    bd[year_col] = pd.to_numeric(bd[year_col], errors="coerce").astype(int)
    keep = keep.merge(bd, on=[cell_col, year_col], how="left")

    never_treated_rows = keep.loc[keep["never_treated"].astype(bool)]
    per_cell_min = never_treated_rows.groupby(cell_col)["nearest_active_pa_border_km"].min()
    banded_controls = per_cell_min[(per_cell_min >= float(min_km)) & (per_cell_min <= float(max_km))].index

    keep["first_treat_for_did"] = treatment_group_for_did(keep[first_treat_col])
    pre_panel = keep[treated_before_panel_col].fillna(False).astype(bool)
    in_window_treated = (keep["first_treat_for_did"] > int(min_year)) & ~pre_panel
    clean_banded_control = keep[cell_col].isin(banded_controls)
    did = keep.loc[in_window_treated | clean_banded_control].copy()
    did.loc[clean_banded_control, "first_treat_for_did"] = 0
    return did.drop(columns=["nearest_active_pa_border_km"])


def build_treated_distance_ring_did_panel(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    min_km: float,
    max_km: float | None,
    cell_col: str,
    year_col: str,
    first_treat_col: str,
    treated_before_panel_col: str,
    outcome_col: str,
    min_year: int,
    max_year: int | None = None,
) -> pd.DataFrame:
    """Buffered DiD panel with a MIN/MAX control-distance ring, sourced from `exposure`'s
    `nearest_treated_distance_km` (computed for every cell, unbounded) rather than
    `build_boundary_band_did_panel`'s `nearest_active_pa_border_km` (notebook 07's border
    table, capped at a 25km candidate-search radius). Needed for any ring that extends past
    that cap -- e.g. a 25-35km bridge band, or an unbounded "35km+" ring matching the main
    buffer's own threshold (pass `max_km=None` for that case). Same treatment-side/control-
    side logic as `build_boundary_band_did_panel` otherwise: treatment is identical to
    `build_stable_buffered_did_panel`, only which never-treated cells count as a clean
    control for this ring changes.

    Boundaries are `(min_km, max_km]` (or `(min_km, inf)` when `max_km` is None), matching
    `build_stable_buffered_did_panel`'s strict `>` convention -- so a 25-35km ring and an
    unbounded 35km+ ring partition the space beyond 25km with no gap or overlap.
    """
    use_cols = [cell_col, year_col, outcome_col, first_treat_col, treated_before_panel_col, "never_treated", "ever_treated"]
    keep = panel[use_cols].copy()
    keep[cell_col] = keep[cell_col].astype("string")
    keep[year_col] = pd.to_numeric(keep[year_col], errors="coerce").astype(int)
    keep = keep[keep[year_col] >= int(min_year)].copy()
    if max_year is not None:
        keep = keep[keep[year_col] <= int(max_year)].copy()

    exp = exposure[[cell_col, year_col, "nearest_treated_distance_km"]].copy()
    exp[cell_col] = exp[cell_col].astype("string")
    exp[year_col] = pd.to_numeric(exp[year_col], errors="coerce").astype(int)
    keep = keep.merge(exp, on=[cell_col, year_col], how="left")

    never_treated_rows = keep.loc[keep["never_treated"].astype(bool)]
    per_cell_min = never_treated_rows.groupby(cell_col)["nearest_treated_distance_km"].min()
    if max_km is None:
        banded_controls = per_cell_min[per_cell_min > float(min_km)].index
    else:
        banded_controls = per_cell_min[(per_cell_min > float(min_km)) & (per_cell_min <= float(max_km))].index

    keep["first_treat_for_did"] = treatment_group_for_did(keep[first_treat_col])
    pre_panel = keep[treated_before_panel_col].fillna(False).astype(bool)
    in_window_treated = (keep["first_treat_for_did"] > int(min_year)) & ~pre_panel
    clean_banded_control = keep[cell_col].isin(banded_controls)
    did = keep.loc[in_window_treated | clean_banded_control].copy()
    did.loc[clean_banded_control, "first_treat_for_did"] = 0
    return did.drop(columns=["nearest_treated_distance_km"])


def cohort_year_support(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    buffer_km: float,
    cell_col: str,
    year_col: str,
    first_treat_col: str,
    treated_col: str,
    treated_before_panel_col: str,
    min_year: int,
) -> pd.DataFrame:
    base = panel[[cell_col, year_col, first_treat_col, treated_col, treated_before_panel_col, "never_treated"]].copy()
    base[cell_col] = base[cell_col].astype("string")
    base[year_col] = pd.to_numeric(base[year_col], errors="coerce").astype(int)
    base = base[base[year_col] >= int(min_year)].copy()
    exp = exposure[[cell_col, year_col, "nearest_treated_distance_km"]].copy()
    exp[cell_col] = exp[cell_col].astype("string")
    exp[year_col] = pd.to_numeric(exp[year_col], errors="coerce").astype(int)
    base = base.merge(exp, on=[cell_col, year_col], how="left")
    base["outside_buffer"] = base["nearest_treated_distance_km"].isna() | (base["nearest_treated_distance_km"] > float(buffer_km))
    base["first_treat_for_did"] = treatment_group_for_did(base[first_treat_col])
    base = base[~base[treated_before_panel_col].fillna(False).astype(bool)].copy()

    cohorts = sorted(g for g in base["first_treat_for_did"].unique() if g > int(min_year))
    years = sorted(base[year_col].unique())
    rows = []
    for g in cohorts:
        treated_ids = base.loc[base["first_treat_for_did"] == g, cell_col].drop_duplicates()
        for t in years:
            if t < min_year:
                continue
            year_df = base[base[year_col] == t]
            control = year_df[
                ((year_df["first_treat_for_did"] == 0) | (year_df["first_treat_for_did"] > t))
                & (year_df[treated_col] == 0)
                & year_df["outside_buffer"]
            ]
            rows.append(
                {
                    "cohort_year": int(g),
                    "year": int(t),
                    "event_time": int(t - g),
                    "n_treated_cells": int(len(treated_ids)),
                    "n_eligible_control_cells": int(control[cell_col].nunique()),
                    "buffer_km": float(buffer_km),
                }
            )
    return pd.DataFrame(rows)


def standardized_mean_differences(
    df: pd.DataFrame,
    *,
    group_col: str,
    covariates: list[str],
    treated_value=1,
) -> pd.DataFrame:
    rows = []
    group = df[group_col] == treated_value
    for cov in covariates:
        x1 = pd.to_numeric(df.loc[group, cov], errors="coerce")
        x0 = pd.to_numeric(df.loc[~group, cov], errors="coerce")
        pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2.0)
        rows.append(
            {
                "covariate": cov,
                "treated_mean": float(x1.mean()),
                "control_mean": float(x0.mean()),
                "standardized_mean_difference": float((x1.mean() - x0.mean()) / pooled) if pooled > 0 else np.nan,
                "treated_missing_share": float(x1.isna().mean()),
                "control_missing_share": float(x0.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def ht_hajek_ring_diagnostics(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    year_col: str,
    ring_col: str,
    reference_label: str,
) -> pd.DataFrame:
    rows = []
    for year, year_df in df.groupby(year_col):
        probs = year_df[ring_col].value_counts(normalize=True, dropna=False)
        ref = year_df[year_df[ring_col] == reference_label].copy()
        if ref.empty:
            continue
        pi_ref = float(probs.get(reference_label, np.nan))
        y_ref = pd.to_numeric(ref[outcome_col], errors="coerce")
        ref_ht_mean = float((y_ref / pi_ref).sum() / len(year_df)) if pi_ref > 0 else np.nan
        ref_hajek_mean = float(y_ref.mean())
        for ring, sub in year_df.groupby(ring_col):
            if ring == reference_label:
                continue
            pi = float(probs.get(ring, np.nan))
            y = pd.to_numeric(sub[outcome_col], errors="coerce")
            ht_mean = float((y / pi).sum() / len(year_df)) if pi > 0 else np.nan
            hajek_mean = float(y.mean())
            rows.append(
                {
                    "year": int(year),
                    "ring": str(ring),
                    "reference_ring": reference_label,
                    "n_ring": int(len(sub)),
                    "n_reference": int(len(ref)),
                    "exposure_probability": pi,
                    "reference_probability": pi_ref,
                    "ht_difference": ht_mean - ref_ht_mean,
                    "hajek_difference": hajek_mean - ref_hajek_mean,
                }
            )
    return pd.DataFrame(rows)


def bartlett_kernel(distance: np.ndarray, cutoff: float) -> np.ndarray:
    scaled = np.asarray(distance, dtype=float) / float(cutoff)
    return np.clip(1.0 - scaled, 0.0, None)


def uniform_kernel(distance: np.ndarray, cutoff: float) -> np.ndarray:
    return (np.asarray(distance, dtype=float) <= float(cutoff)).astype(float)


def empirical_variogram(
    df: pd.DataFrame,
    *,
    value_col: str,
    x_col: str = "centroid_x_m",
    y_col: str = "centroid_y_m",
    max_lag_km: float = 100.0,
    n_bins: int = 20,
    chunk_size: int = 2000,
    max_pairs: int = 20_000_000,
    max_points: int | None = 200_000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Empirical semivariogram: gamma(h) = mean over pairs at distance h of 0.5*(Z_i - Z_j)^2.

    Same streaming-KDTree-neighborhood approach as spatial_hac_variance -- never
    materializes a full N x N pairwise-distance matrix, bounded by max_lag_km with a
    max_pairs safety cap. Estimating a variogram doesn't need every unit, so if `df` has
    more than `max_points` rows a uniform random subsample is taken first; the decay curve
    this recovers is the same, just with tighter compute.

    Returns one row per distance bin: `lag_km` (bin center), `semivariance`, `n_pairs`.
    Intended uses in this project: (1) a purely descriptive check of how far spatial
    dependence in forest loss extends, on a variable like forest_loss_pre_mean_m2 or
    loss_m2; (2) validating HAC_CUTOFF_KM/HAC_KERNEL by variogramming the actual per-unit
    influence contributions (psi) spatial_hac_variance sums over for a given ATT(g,t) cell
    -- if psi's empirical range is much shorter or longer than HAC_CUTOFF_KM, or its decay
    shape doesn't match the Bartlett/uniform kernel currently used, that's a concrete basis
    to adjust those constants. These two targets answer different questions and are not
    interchangeable: raw outcome levels reflect shared ecology/climate/regional policy and
    will show strong, long-range correlation almost everywhere, which is NOT the same thing
    as correlation in estimation error.
    """
    d = df[[x_col, y_col, value_col]].dropna().copy()
    if max_points is not None and len(d) > max_points:
        d = d.sample(n=max_points, random_state=random_state)
    coords = d[[x_col, y_col]].to_numpy(dtype=float)
    z = d[value_col].to_numpy(dtype=float)
    max_lag_m = float(max_lag_km) * 1000.0
    tree = cKDTree(coords)

    bin_edges = np.linspace(0.0, max_lag_m, int(n_bins) + 1)
    sq_diff_sum = np.zeros(n_bins)
    pair_count = np.zeros(n_bins, dtype=np.int64)

    n = len(d)
    n_pairs_used = 0
    truncated = False
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        neighborhoods = tree.query_ball_point(coords[start:stop], r=max_lag_m)
        for local_i, nbrs in enumerate(neighborhoods):
            i = start + local_i
            nbr_idx = np.asarray([j for j in nbrs if j > i], dtype=int)
            if nbr_idx.size == 0:
                continue
            if n_pairs_used + nbr_idx.size > int(max_pairs):
                keep = max(int(max_pairs) - n_pairs_used, 0)
                nbr_idx = nbr_idx[:keep]
                truncated = True
            if nbr_idx.size == 0:
                break
            dist = np.sqrt(((coords[nbr_idx] - coords[i]) ** 2).sum(axis=1))
            bins = np.digitize(dist, bin_edges) - 1
            valid = (bins >= 0) & (bins < n_bins)
            sq_diff = 0.5 * (z[i] - z[nbr_idx]) ** 2
            np.add.at(sq_diff_sum, bins[valid], sq_diff[valid])
            np.add.at(pair_count, bins[valid], 1)
            n_pairs_used += int(nbr_idx.size)
            if truncated:
                break
        if truncated:
            break

    if truncated:
        print(f"WARNING: empirical_variogram hit max_pairs={max_pairs:,} before covering all "
              f"{n:,} points -- increase max_pairs, reduce max_lag_km, or reduce max_points.")

    bin_centers_km = (bin_edges[:-1] + bin_edges[1:]) / 2.0 / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        semivariance = np.where(pair_count > 0, sq_diff_sum / pair_count, np.nan)

    return pd.DataFrame({"lag_km": bin_centers_km, "semivariance": semivariance, "n_pairs": pair_count})


def fit_spherical_variogram(lag_km, semivariance, *, n_pairs=None) -> dict:
    """Fit the classic nugget/sill/range spherical model to an empirical variogram.

    model(h) = nugget + (sill - nugget) * (1.5*(h/range) - 0.5*(h/range)**3) for h <= range,
               sill for h > range -- the textbook geostatistical approximation (Cressie 1993),
               used here instead of connecting the raw bin dots so the figure reports a single
               interpretable range (the lag at which the fitted curve first reaches the sill)
               rather than requiring a reader to eyeball where the scatter flattens out.

    `n_pairs`, if given, weights the least-squares fit by pair count per bin (bins built from
    very few pairs -- typically the longest lags -- are noisier and should move the fit less).

    Returns a dict with `nugget`, `sill`, `range_km`, `r_squared`, and `predict` (a callable
    mapping an array of lags in km to fitted semivariance).
    """
    from scipy.optimize import curve_fit

    lag_km = np.asarray(lag_km, dtype=float)
    semivariance = np.asarray(semivariance, dtype=float)
    mask = np.isfinite(lag_km) & np.isfinite(semivariance)
    if n_pairs is not None:
        n_pairs = np.asarray(n_pairs, dtype=float)
        mask &= np.isfinite(n_pairs) & (n_pairs > 0)
    lag_km, semivariance = lag_km[mask], semivariance[mask]

    def spherical(h, nugget, sill, rng):
        h = np.asarray(h, dtype=float)
        ratio = np.clip(h / rng, 0.0, 1.0)
        return nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3)

    sill_guess = float(semivariance.max())
    nugget_guess = float(max(semivariance.min(), 0.0))
    past_sill = lag_km[semivariance >= 0.95 * sill_guess]
    range_guess = float(past_sill.min()) if past_sill.size else float(lag_km.max() / 2.0)
    p0 = [nugget_guess, sill_guess, max(range_guess, 1e-3)]
    bounds = ([0.0, 0.0, 1e-6], [sill_guess * 1.5, sill_guess * 3.0, lag_km.max() * 3.0])

    sigma = 1.0 / np.sqrt(n_pairs[mask]) if n_pairs is not None else None
    params, _ = curve_fit(spherical, lag_km, semivariance, p0=p0, bounds=bounds, sigma=sigma, maxfev=20000)
    nugget, sill, rng = (float(p) for p in params)

    fitted = spherical(lag_km, nugget, sill, rng)
    ss_res = float(np.sum((semivariance - fitted) ** 2))
    ss_tot = float(np.sum((semivariance - semivariance.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "nugget": nugget,
        "sill": sill,
        "range_km": rng,
        "r_squared": r_squared,
        "predict": lambda h: spherical(h, nugget, sill, rng),
    }


def spatial_hac_variance(
    influence_df: pd.DataFrame,
    *,
    x_col: str = "centroid_x_m",
    y_col: str = "centroid_y_m",
    psi_col: str = "psi",
    cutoff_km: float = 25.0,
    kernel: str = "bartlett",
    chunk_size: int = 5000,
    max_pairs: int = 50_000_000,
) -> dict:
    """Compute a Conley-style spatial HAC variance from cell-level influence values.

    The input should already be collapsed to one row per spatial unit for the
    estimate being evaluated. This routine streams KDTree neighborhoods in
    chunks so it does not materialize a full all-pairs matrix. It still can be
    expensive at 1km resolution with large cutoffs, so max_pairs is an explicit
    safety guard.
    """
    df = influence_df[[x_col, y_col, psi_col]].dropna().copy()
    if df.empty:
        return {
            "variance": np.nan,
            "standard_error": np.nan,
            "n_units": 0,
            "cutoff_km": float(cutoff_km),
            "kernel": kernel,
            "n_pairs_used": 0,
            "truncated": False,
        }

    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    psi = df[psi_col].to_numpy(dtype=float)
    cutoff_m = float(cutoff_km) * 1000.0
    tree = cKDTree(coords)

    if kernel == "bartlett":
        kernel_func = bartlett_kernel
    elif kernel == "uniform":
        kernel_func = uniform_kernel
    else:
        raise ValueError("kernel must be either 'bartlett' or 'uniform'.")

    total = float(np.sum(psi * psi))
    n_pairs_used = 0
    truncated = False
    n = len(df)

    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        neighborhoods = tree.query_ball_point(coords[start:stop], r=cutoff_m)
        for local_i, nbrs in enumerate(neighborhoods):
            i = start + local_i
            nbr_idx = np.asarray([j for j in nbrs if j > i], dtype=int)
            if nbr_idx.size == 0:
                continue
            if n_pairs_used + nbr_idx.size > int(max_pairs):
                keep = max(int(max_pairs) - n_pairs_used, 0)
                nbr_idx = nbr_idx[:keep]
                truncated = True
            if nbr_idx.size == 0:
                break
            dist = np.sqrt(((coords[nbr_idx] - coords[i]) ** 2).sum(axis=1))
            weights = kernel_func(dist, cutoff_m)
            total += float(2.0 * np.sum(weights * psi[i] * psi[nbr_idx]))
            n_pairs_used += int(nbr_idx.size)
            if truncated:
                break
        if truncated:
            break

    variance = max(total, 0.0)
    return {
        "variance": variance,
        "standard_error": float(np.sqrt(variance)),
        "n_units": int(n),
        "cutoff_km": float(cutoff_km),
        "kernel": kernel,
        "n_pairs_used": int(n_pairs_used),
        "truncated": bool(truncated),
    }


def split_panel_by_year(panel: pd.DataFrame, year_col: str) -> dict:
    """{year: sub-dataframe} split, built once so did_influence_for_group_time and
    dr_did_influence_for_group_time can look up a year's rows directly instead of
    re-filtering the full (potentially multi-million-row) panel by year on every single
    (g,t) call -- these functions are typically called 1000+ times over a full notebook run
    (every cohort x post-period x spec x band), so the repeated boolean-mask scan was
    dominating runtime independent of the actual model-fitting cost.
    """
    return {int(y): sub for y, sub in panel.groupby(year_col)}


def did_influence_for_group_time(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    outcome_col: str,
    group_col: str,
    group_year: int,
    time_year: int,
    base_year: int,
    panel_by_year: dict | None = None,
) -> tuple[float, pd.DataFrame]:
    """No-covariate 2x2 DID influence contributions for one ATT(g,t).

    This is intended as the HAC inference layer for the no-covariate
    specification. The point estimate can be compared with package output.

    `panel_by_year`, if supplied (see split_panel_by_year), is used instead of filtering
    `panel` directly -- same result, much less repeated work across many calls. Falls back
    to filtering `panel` when not supplied, so existing callers are unaffected.
    """
    use_years = [int(base_year), int(time_year)]
    cols = [cell_col, year_col, outcome_col, group_col]
    if panel_by_year is not None:
        frames = [panel_by_year[y][cols] for y in use_years if y in panel_by_year]
        if len(frames) < len(use_years):
            return np.nan, pd.DataFrame()
        year_rows = pd.concat(frames, ignore_index=True)
    else:
        year_rows = panel.loc[panel[year_col].isin(use_years), cols]
    wide = (
        year_rows
        .pivot_table(index=[cell_col, group_col], columns=year_col, values=outcome_col, aggfunc="mean")
        .reset_index()
    )
    if int(base_year) not in wide.columns or int(time_year) not in wide.columns:
        return np.nan, pd.DataFrame()
    wide = wide.dropna(subset=[int(base_year), int(time_year)]).copy()
    wide["delta_y"] = wide[int(time_year)] - wide[int(base_year)]
    treated_mask = wide[group_col] == int(group_year)
    control_mask = wide[group_col] == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    if n_treated == 0 or n_control == 0:
        return np.nan, pd.DataFrame()

    treated_mean = wide.loc[treated_mask, "delta_y"].mean()
    control_mean = wide.loc[control_mask, "delta_y"].mean()
    estimate = float(treated_mean - control_mean)

    wide["psi"] = 0.0
    wide.loc[treated_mask, "psi"] = (wide.loc[treated_mask, "delta_y"] - treated_mean) / n_treated
    wide.loc[control_mask, "psi"] = -(wide.loc[control_mask, "delta_y"] - control_mean) / n_control
    influence = wide.loc[treated_mask | control_mask, [cell_col, "psi"]].merge(
        centroids[[cell_col, "centroid_x_m", "centroid_y_m"]],
        on=cell_col,
        how="left",
    )
    return estimate, influence


def fit_cohort_propensity_score(
    panel: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    group_col: str,
    covariates: list[str],
    group_year: int,
    base_year: int,
    cv: int = 5,
    trim: float = 0.02,
) -> dict:
    """Fit ONE standardized, cross-validated-regularized propensity-score model per cohort.

    The propensity score only needs to separate treated cohort `group_year` from the
    never-treated pool as of `base_year` -- it does not depend on which post-period t an
    ATT(g,t) call is evaluating. Fitting it once per cohort and reusing it (via
    dr_did_influence_for_group_time's `ps_model` argument) across every t for that cohort,
    instead of refitting a fresh unregularized logistic regression from scratch inside every
    single (g,t) cell, matters for two reasons: (1) it multiplies the effective fitting
    sample by however many post-periods the cohort has, which directly reduces small-sample
    separation risk; (2) a sparse, regionally-clustered covariate (e.g. coca_pre_mean_ha,
    genuinely observed for ~12% of cells) is much more likely to look "perfectly separating"
    by chance in a small per-cell sample than in the larger pooled-per-cohort one. Combined
    with standardizing covariates and choosing the L2 penalty via cross-validation (rather
    than the near-unregularized C=1e6 fixed fit this replaces), this is what keeps a sparse
    covariate like coca in the model without letting a handful of cells drive extreme
    propensity scores. Explicit trimming at `trim` (Crump-style) is applied by the caller
    using the returned model, not here -- this function only fits and reports how many units
    WOULD be trimmed, so the trim rate is visible even when trimming isn't yet applied.

    Returns a dict: `scaler`/`model` (None if the cohort can't support a fit -- too few
    control units, or only one class present), `n_treated`, `n_control`, `n_extreme_trimmed`,
    `share_trimmed`, `trim`.
    """
    cols = [cell_col, group_col] + list(covariates)
    cross = (
        panel.loc[panel[year_col] == int(base_year), cols]
        .drop_duplicates(subset=[cell_col])
        .dropna(subset=list(covariates))
        .copy()
    )
    treated_mask = cross[group_col] == int(group_year)
    control_mask = cross[group_col] == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    empty_result = {
        "scaler": None,
        "model": None,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_extreme_trimmed": np.nan,
        "share_trimmed": np.nan,
        "trim": float(trim),
    }
    if n_treated == 0 or n_control < len(covariates) + 2:
        return empty_result

    use = cross.loc[treated_mask | control_mask].copy()
    d = use[group_col].eq(int(group_year)).astype(float).to_numpy()
    if np.unique(d).size < 2:
        return empty_result
    x = use[list(covariates)].to_numpy(dtype=float)

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    x_std = scaler.transform(x)
    minority_class_n = int(np.bincount(d.astype(int)).min())
    n_splits = min(int(cv), minority_class_n)
    if n_splits < 2:
        # Too few units in the smaller class for cross-validated regularization strength
        # selection -- fall back to a fixed, still-real (not 1e6) penalty rather than CV.
        model = LogisticRegression(C=1.0, max_iter=2000).fit(x_std, d)
    else:
        model = LogisticRegressionCV(Cs=10, cv=n_splits, scoring="neg_log_loss", max_iter=2000).fit(x_std, d)

    p_hat = model.predict_proba(x_std)[:, 1]
    n_extreme = int(np.sum((p_hat < float(trim)) | (p_hat > 1.0 - float(trim))))
    return {
        "scaler": scaler,
        "model": model,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_extreme_trimmed": n_extreme,
        "share_trimmed": n_extreme / len(p_hat) if len(p_hat) else np.nan,
        "trim": float(trim),
    }


def dr_did_influence_for_group_time(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    outcome_col: str,
    group_col: str,
    covariates: list[str],
    group_year: int,
    time_year: int,
    base_year: int,
    ps_model: dict | None = None,
    ps_trim: float = 0.02,
    ps_max_weight: float = 100.0,
    panel_by_year: dict | None = None,
) -> tuple[float, pd.DataFrame]:
    """Doubly-robust (Sant'Anna and Zhao 2020) 2x2 ATT(g,t) influence contributions.

    This is the covariate-adjusted counterpart to did_influence_for_group_time, built
    so the same spatial_hac_variance() layer can be applied to the dr_geography* specs.
    It refits its own outcome-regression per ATT(g,t) cell (delta_y genuinely depends on
    time_year, so this one can't be cached across t the way the propensity score is). The
    outcome regression is a standardized, cross-validated-regularized Ridge fit (control
    units only) rather than plain OLS -- OLS is fit on a large control pool but then
    *evaluated* (predicted) at the treated cohort's own covariate values, which can sit
    outside the region the control-fitted model has real support for; unregularized OLS
    extrapolates linearly and unboundedly into that region, and a handful of treated units
    with atypical covariates is enough to produce wildly wrong predictions that dominate the
    tau1 term below. Ridge shrinks coefficients toward zero, which directly damps how far a
    prediction can swing under exactly this kind of extrapolation. RidgeCV's default
    (`cv=None`) uses the efficient closed-form leave-one-out formula, so this adds negligible
    runtime over plain OLS despite the added robustness.

    The propensity score, which does NOT depend on time_year, should be fit ONCE per cohort
    via fit_cohort_propensity_score and passed in as `ps_model` -- if omitted, this falls
    back to an unregularized, per-cell logistic fit (the original, more separation-prone
    behavior) purely for backward compatibility. Either way, the fitted/cached propensity is
    trimmed to [ps_trim, 1-ps_trim] before forming weights, then forms the Sant'Anna-Zhao
    "traditional" DR-DID estimator

        tau_dr = E_n[w1*(delta_y - m0_hat)] / E_n[w1]  -  E_n[w0*(delta_y - m0_hat)] / E_n[w0]

    with w1 = D, w0 = phat(X)*(1-D)/(1-phat(X)).

    Approximation note: the returned psi is the *plug-in* influence function -- it treats
    m0_hat and phat as fixed (i.e. it is the influence function of the two ratio estimators
    E_n[w1*(.)]/E_n[w1] and E_n[w0*(.)]/E_n[w0] given the fitted nuisance functions) and does
    NOT include the Sant'Anna and Zhao (2020, Theorem 1) asymptotic correction terms for
    outcome-regression and propensity-score estimation uncertainty (their M1/M2 terms). That
    correction requires the score/Hessian of both nuisance fits and was not re-derived here.
    In exchange this is exactly verifiable against the DR point estimate formula above. It is
    expected to be a mild UNDERESTIMATE of the true DR-DID variance -- a similar direction of
    bias to ignoring first-step estimation uncertainty in other two-step GMM/M-estimation
    settings -- so treat the resulting HAC SE as a lower bound, not the final word, on the
    dr_geography* specifications' true spatial-HAC uncertainty.

    Interpretation note: regularizing BOTH nuisance models (this function's outcome
    regression, and fit_cohort_propensity_score's propensity fit) shrinks each toward zero
    relative to an unregularized fit, attenuating the DR estimate toward whichever component
    would otherwise dominate; `ps_trim` additionally restricts the estimate to a
    common-support subpopulation (whatever share of units would otherwise have phat outside
    [ps_trim, 1-ps_trim]). All three trade a small, documented bias for the stability an
    unregularized, untrimmed per-cell fit lacks under sparse, regionally-clustered, or
    otherwise poorly-supported covariates.
    """
    use_years = [int(base_year), int(time_year)]
    cols = [cell_col, year_col, outcome_col, group_col] + list(covariates)
    if panel_by_year is not None:
        frames = [panel_by_year[y][cols] for y in use_years if y in panel_by_year]
        if len(frames) < len(use_years):
            return np.nan, pd.DataFrame()
        year_rows = pd.concat(frames, ignore_index=True)
    else:
        year_rows = panel.loc[panel[year_col].isin(use_years), cols]
    wide = (
        year_rows
        .drop_duplicates(subset=[cell_col, year_col])
        .pivot_table(index=[cell_col, group_col] + list(covariates), columns=year_col, values=outcome_col, aggfunc="mean")
        .reset_index()
    )
    if int(base_year) not in wide.columns or int(time_year) not in wide.columns:
        return np.nan, pd.DataFrame()
    wide = wide.dropna(subset=[int(base_year), int(time_year)] + list(covariates)).copy()
    wide["delta_y"] = wide[int(time_year)] - wide[int(base_year)]
    treated_mask = wide[group_col] == int(group_year)
    control_mask = wide[group_col] == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    if n_treated == 0 or n_control == 0:
        return np.nan, pd.DataFrame()

    use = wide.loc[treated_mask | control_mask].copy()
    d = use[group_col].eq(int(group_year)).astype(float).to_numpy()
    x = use[list(covariates)].to_numpy(dtype=float)
    delta_y = use["delta_y"].to_numpy(dtype=float)
    n = len(use)
    if n_control < len(covariates) + 2 or n_treated < 1:
        return np.nan, pd.DataFrame()

    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    or_scaler = StandardScaler().fit(x[d == 0.0])
    x_std = or_scaler.transform(x)
    or_model = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(x_std[d == 0.0], delta_y[d == 0.0])
    m0_hat = or_model.predict(x_std)

    if ps_model is not None and ps_model.get("model") is not None:
        scaler, ps_fit = ps_model["scaler"], ps_model["model"]
        p_hat_raw = ps_fit.predict_proba(scaler.transform(x))[:, 1]
    else:
        if np.unique(d).size < 2:
            return np.nan, pd.DataFrame()
        ps_fit = LogisticRegression(C=1e6, max_iter=2000).fit(x, d)
        p_hat_raw = ps_fit.predict_proba(x)[:, 1]
    p_hat = np.clip(p_hat_raw, float(ps_trim), 1.0 - float(ps_trim))

    w1 = d
    w0 = p_hat * (1.0 - d) / (1.0 - p_hat)
    w0 = np.minimum(w0, float(ps_max_weight))

    e_w1 = w1.mean()
    e_w0 = w0.mean()
    if e_w1 <= 0 or e_w0 <= 0:
        return np.nan, pd.DataFrame()

    resid = delta_y - m0_hat
    tau1 = float((w1 * resid).mean() / e_w1)
    tau0 = float((w0 * resid).mean() / e_w0)
    estimate = tau1 - tau0

    # The extra factor of n below (on top of e_w1/e_w0, the MEAN weight) converts the
    # per-observation moment contribution into the same "already normalized by effective
    # sample size" convention did_influence_for_group_time uses (there, psi is divided by
    # the raw treated/control COUNT, not by n_treated/n) -- so spatial_hac_variance's
    # Var_hat = sum psi_i^2 + ... reduces to the usual Var(sample mean) formula in the
    # no-covariate limit (w1=D, w0=1-D), and to the analogous ratio-estimator variance here.
    psi1 = (w1 * resid - w1 * tau1) / (e_w1 * n)
    psi0 = (w0 * resid - w0 * tau0) / (e_w0 * n)
    use["psi"] = psi1 - psi0

    influence = use.loc[:, [cell_col, "psi"]].merge(
        centroids[[cell_col, "centroid_x_m", "centroid_y_m"]],
        on=cell_col,
        how="left",
    )
    return float(estimate), influence


def boundary_cohort_ids(
    boundary_distance: pd.DataFrame,
    *,
    cell_col: str,
    cohort_year: int,
    interior_max_km: float,
    exterior_min_km: float,
    exterior_max_km: float,
) -> tuple[set, set]:
    """Near-border treated and near-border never-treated cell-id sets for one cohort.

    Split out from boundary_did_influence so a caller looping many event times for the
    same cohort computes this -- and filters the (potentially large) outcome panel down to
    these ids -- ONCE, rather than repeating both an expensive boundary_distance filter and
    a full-panel isin() once per event time.
    """
    cand = boundary_distance.loc[boundary_distance["cohort_year"] == int(cohort_year)]
    treated_ids = set(
        cand.loc[
            cand["is_cohort_treated"].astype(bool) & (cand["distance_to_boundary_km"] <= float(interior_max_km)),
            cell_col,
        ]
    )
    control_ids = set(
        cand.loc[
            (~cand["is_cohort_treated"].astype(bool))
            & (cand["never_treated"] == 1)
            & (cand["distance_to_boundary_km"] >= float(exterior_min_km))
            & (cand["distance_to_boundary_km"] <= float(exterior_max_km)),
            cell_col,
        ]
    )
    return treated_ids, control_ids


def boundary_did_influence(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    outcome_col: str,
    treated_ids: set,
    control_ids: set,
    base_year: int,
    time_year: int,
) -> tuple[float, pd.DataFrame]:
    """2x2 DID influence contributions: near-border TREATED cells of one cohort vs.
    near-border NEVER-TREATED cells (id sets from boundary_cohort_ids) instead of a
    panel-wide treatment-group column.

    This is the "inside vs. outside the PA border" design -- unlike
    did_influence_for_group_time (which compares a cohort's full treated set to a clean
    far-away control set), the treated arm here is itself restricted to cells within
    interior_max_km of that cohort's own dissolved boundary, contrasted against
    never-treated cells in the [exterior_min_km, exterior_max_km] donut just outside it.
    Same psi convention as did_influence_for_group_time, so spatial_hac_variance applies
    unchanged. `panel` may already be pre-filtered to treated_ids | control_ids by the
    caller (recommended when looping many event times per cohort) -- filtering again here
    is a no-op in that case.
    """
    if not treated_ids or not control_ids:
        return np.nan, pd.DataFrame()

    use_ids = treated_ids | control_ids
    use_years = [int(base_year), int(time_year)]
    wide = (
        panel.loc[panel[cell_col].isin(use_ids) & panel[year_col].isin(use_years), [cell_col, year_col, outcome_col]]
        .pivot_table(index=cell_col, columns=year_col, values=outcome_col, aggfunc="mean")
        .reset_index()
    )
    if int(base_year) not in wide.columns or int(time_year) not in wide.columns:
        return np.nan, pd.DataFrame()
    wide = wide.dropna(subset=[int(base_year), int(time_year)]).copy()
    wide["delta_y"] = wide[int(time_year)] - wide[int(base_year)]
    treated_mask = wide[cell_col].isin(treated_ids)
    control_mask = wide[cell_col].isin(control_ids)
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    if n_treated == 0 or n_control == 0:
        return np.nan, pd.DataFrame()

    treated_mean = wide.loc[treated_mask, "delta_y"].mean()
    control_mean = wide.loc[control_mask, "delta_y"].mean()
    estimate = float(treated_mean - control_mean)

    wide["psi"] = 0.0
    wide.loc[treated_mask, "psi"] = (wide.loc[treated_mask, "delta_y"] - treated_mean) / n_treated
    wide.loc[control_mask, "psi"] = -(wide.loc[control_mask, "delta_y"] - control_mean) / n_control
    # is_treated is carried alongside psi so callers can recover n_treated/n_control from the
    # returned influence frame -- psi's sign reflects each unit's deviation from its OWN
    # group's mean, not which group it's in, so it can't be used to infer group membership.
    wide["is_treated"] = treated_mask
    influence = wide.loc[treated_mask | control_mask, [cell_col, "psi", "is_treated"]].merge(
        centroids[[cell_col, "centroid_x_m", "centroid_y_m"]],
        on=cell_col,
        how="left",
    )
    return estimate, influence


def hac_for_group_time_effects(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    outcome_col: str,
    group_col: str,
    effects: pd.DataFrame,
    cutoff_km: float = 25.0,
    kernel: str = "bartlett",
    max_pairs: int = 50_000_000,
    anticipation: int = 0,
    covariates: list[str] | None = None,
    ps_trim: float = 0.02,
    ps_cv: int = 5,
    panel_by_year: dict | None = None,
) -> pd.DataFrame:
    """Spatial-HAC inference layer for ATT(g,t) -- and, for the dr_geography* specs, the
    source of the ATT(g,t) point estimate itself (see aggregate_group_time_atts, which rolls
    these up into the reported overall ATT the same way for every spec).

    `anticipation` shifts the base (pre-treatment reference) period back by that many
    years, mirroring csdid's ATTgt(anticipation=...) -- pass the same value used to fit the
    no-covariate specs' point estimates so the recomputed 2x2 DID here (and its HAC SE) is
    anchored to the same base period, not silently defaulting back to group_year - 1.

    `covariates`, when non-empty, switches the per-(g,t) influence function from the plain
    2x2 diff-in-means (did_influence_for_group_time) to the doubly-robust plug-in estimator
    (dr_did_influence_for_group_time) -- see that function's docstring for the DR variance
    approximation this implies. Pass None/[] for the no-covariate specs.

    For the DR case, the propensity score is fit ONCE per cohort (fit_cohort_propensity_score,
    standardized + cross-validated regularization) and reused across every t for that cohort,
    rather than refit per (g,t) cell -- see that function's docstring for why. `ps_trim` sets
    the common-support restriction applied on top; `ps_n_control`/`ps_share_trimmed` are
    reported per row so trimming is never silent.
    """
    columns = [
        "group",
        "time",
        "event_time",
        "att_no_covariate_recomputed",
        "spatial_hac_se",
        "hac_n_units",
        "hac_n_pairs_used",
        "hac_truncated",
        "hac_cutoff_km",
        "hac_kernel",
        "anticipation",
        "ps_n_control",
        "ps_share_trimmed",
    ]
    ps_models: dict[int, dict] = {}
    if covariates:
        for group_year in sorted(int(g) for g in effects["group"].unique()):
            base_year = group_year - 1 - int(anticipation)
            ps_models[group_year] = fit_cohort_propensity_score(
                panel,
                cell_col=cell_col,
                year_col=year_col,
                group_col=group_col,
                covariates=covariates,
                group_year=group_year,
                base_year=base_year,
                cv=ps_cv,
                trim=ps_trim,
            )
    rows = []
    for row in effects.itertuples(index=False):
        group_year = int(getattr(row, "group"))
        time_year = int(getattr(row, "time"))
        base_year = group_year - 1 - int(anticipation)
        ps_diag = {"n_control": np.nan, "share_trimmed": np.nan}
        if covariates:
            ps_diag = ps_models.get(group_year, ps_diag)
            estimate, influence = dr_did_influence_for_group_time(
                panel,
                centroids,
                cell_col=cell_col,
                year_col=year_col,
                outcome_col=outcome_col,
                group_col=group_col,
                covariates=covariates,
                group_year=group_year,
                time_year=time_year,
                base_year=base_year,
                ps_model=ps_models.get(group_year),
                ps_trim=ps_trim,
                panel_by_year=panel_by_year,
            )
        else:
            estimate, influence = did_influence_for_group_time(
                panel,
                centroids,
                cell_col=cell_col,
                year_col=year_col,
                outcome_col=outcome_col,
                group_col=group_col,
                group_year=group_year,
                time_year=time_year,
                base_year=base_year,
                panel_by_year=panel_by_year,
            )
        if influence.empty:
            hac = {
                "standard_error": np.nan,
                "n_units": 0,
                "n_pairs_used": 0,
                "truncated": False,
            }
        else:
            hac = spatial_hac_variance(
                influence,
                cutoff_km=cutoff_km,
                kernel=kernel,
                max_pairs=max_pairs,
            )
        rows.append(
            {
                "group": group_year,
                "time": time_year,
                "event_time": time_year - group_year,
                "att_no_covariate_recomputed": estimate,
                "spatial_hac_se": hac["standard_error"],
                "hac_n_units": hac["n_units"],
                "hac_n_pairs_used": hac["n_pairs_used"],
                "hac_truncated": hac["truncated"],
                "hac_cutoff_km": cutoff_km,
                "hac_kernel": kernel,
                "anticipation": int(anticipation),
                "ps_n_control": ps_diag.get("n_control", np.nan),
                "ps_share_trimmed": ps_diag.get("share_trimmed", np.nan),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def spatial_hac_from_cached_influence(
    cached: dict,
    *,
    cutoff_km: float = 25.0,
    kernel: str = "bartlett",
    max_pairs: int = 50_000_000,
    anticipation: int = 0,
) -> pd.DataFrame:
    """Spatial-HAC inference layer for ATT(g,t), reusing influence functions the caller
    already computed -- e.g. a point-estimate loop that called dr_did_influence_for_group_time
    once per (g,t) cell and kept the influence DataFrame around -- instead of recomputing them
    from scratch the way hac_for_group_time_effects does.

    This exists because dr_did_influence_for_group_time (outcome-regression + propensity-
    score fit) is the single most expensive call in the pipeline, and it was being run TWICE
    per (g,t) cell for the dr_geography* specs: once for the point estimate (cell 9's
    fit_dr_grid) and again for the HAC variance (hac_for_group_time_effects, cell 11/21).
    Caching the (estimate, influence) pair from the first pass and feeding it here instead of
    calling hac_for_group_time_effects a second time removes that duplication entirely, with
    no change to any reported number -- this function does no model fitting, it only runs
    spatial_hac_variance() over influence DataFrames that already exist.

    `cached` maps (group, time) -> dict with keys 'estimate', 'influence' (the DataFrame
    did_influence_for_group_time/dr_did_influence_for_group_time returned, or None/empty if
    that cell had no valid influence), and optionally 'ps_n_control'/'ps_share_trimmed' (the
    per-cohort propensity diagnostics from fit_cohort_propensity_score, carried through so
    this reports the same diagnostics hac_for_group_time_effects would have).
    """
    columns = [
        "group",
        "time",
        "event_time",
        "att_no_covariate_recomputed",
        "spatial_hac_se",
        "hac_n_units",
        "hac_n_pairs_used",
        "hac_truncated",
        "hac_cutoff_km",
        "hac_kernel",
        "anticipation",
        "ps_n_control",
        "ps_share_trimmed",
    ]
    rows = []
    for (group_year, time_year), entry in cached.items():
        influence = entry.get("influence")
        estimate = entry.get("estimate", np.nan)
        if influence is None or influence.empty:
            hac = {"standard_error": np.nan, "n_units": 0, "n_pairs_used": 0, "truncated": False}
        else:
            hac = spatial_hac_variance(influence, cutoff_km=cutoff_km, kernel=kernel, max_pairs=max_pairs)
        rows.append(
            {
                "group": int(group_year),
                "time": int(time_year),
                "event_time": int(time_year) - int(group_year),
                "att_no_covariate_recomputed": estimate,
                "spatial_hac_se": hac["standard_error"],
                "hac_n_units": hac["n_units"],
                "hac_n_pairs_used": hac["n_pairs_used"],
                "hac_truncated": hac["truncated"],
                "hac_cutoff_km": cutoff_km,
                "hac_kernel": kernel,
                "anticipation": int(anticipation),
                "ps_n_control": entry.get("ps_n_control", np.nan),
                "ps_share_trimmed": entry.get("ps_share_trimmed", np.nan),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_group_time_atts(
    gt: pd.DataFrame,
    n_treated_by_group: dict,
    *,
    anticipation: int = 0,
    att_col: str = "att",
    group_col: str = "group",
    time_col: str = "time",
) -> dict:
    """Cohort-size-weighted "simple" aggregation of ATT(g,t) into one overall ATT.

    Matches the weighting convention both diff_diff (staggered_aggregation._aggregate_simple:
    weight = n_treated per cohort) and csdid (aggte_fnc.compute_aggte: weight = per-unit
    P(G=g) share) use for their own overall-ATT figure -- applying it ourselves, identically,
    to every ROBUSTNESS_SPECS grid regardless of which estimator produced each cell's point
    estimate (csdid for the no-covariate specs, the in-house DR plug-in for dr_geography*), so
    the four specs' overall numbers are comparable for a reason beyond coincidence: they're
    rolled up the exact same way. Only post-treatment cells (time >= group - anticipation) are
    included, matching both packages' "simple" aggregation.

    Deliberately does NOT compute an overall SE -- the notebook's spatial-HAC layer
    (combine_se_rms over post-treatment spatial_hac_se) is the intended, already-uniform SE
    source across all four specs; this function only standardizes the point-estimate rollup.
    """
    df = gt[[group_col, time_col, att_col]].dropna().copy()
    df[group_col] = pd.to_numeric(df[group_col], errors="coerce").astype(int)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype(int)
    df = df.loc[df[time_col] >= (df[group_col] - int(anticipation))]
    if df.empty:
        return {"overall_att": np.nan, "n_cells": 0}
    weights = df[group_col].map(n_treated_by_group).astype(float)
    valid = weights.notna() & (weights > 0)
    df, weights = df.loc[valid], weights.loc[valid]
    if df.empty:
        return {"overall_att": np.nan, "n_cells": 0}
    overall_att = float(np.sum(weights.to_numpy() * df[att_col].to_numpy()) / np.sum(weights.to_numpy()))
    return {"overall_att": overall_att, "n_cells": int(len(df))}


def hac_for_hajek_ring_contrasts(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cell_col: str,
    year_col: str,
    outcome_col: str,
    ring_col: str,
    reference_label: str,
    cutoff_km: float = 25.0,
    kernel: str = "bartlett",
    max_pairs: int = 50_000_000,
) -> pd.DataFrame:
    rows = []
    centroid_cols = [cell_col, "centroid_x_m", "centroid_y_m"]
    for (year, ring), sub in df[df[ring_col] != reference_label].groupby([year_col, ring_col]):
        ref = df[(df[year_col] == year) & (df[ring_col] == reference_label)].copy()
        ring_df = sub.copy()
        if ref.empty or ring_df.empty:
            continue
        ring_mean = pd.to_numeric(ring_df[outcome_col], errors="coerce").mean()
        ref_mean = pd.to_numeric(ref[outcome_col], errors="coerce").mean()
        estimate = float(ring_mean - ref_mean)
        ring_df["psi"] = (pd.to_numeric(ring_df[outcome_col], errors="coerce") - ring_mean) / len(ring_df)
        ref["psi"] = -(pd.to_numeric(ref[outcome_col], errors="coerce") - ref_mean) / len(ref)
        influence = pd.concat(
            [ring_df[[cell_col, "psi"]], ref[[cell_col, "psi"]]],
            ignore_index=True,
        ).merge(centroids[centroid_cols], on=cell_col, how="left")
        hac = spatial_hac_variance(
            influence,
            cutoff_km=cutoff_km,
            kernel=kernel,
            max_pairs=max_pairs,
        )
        rows.append(
            {
                "year": int(year),
                "ring": str(ring),
                "reference_ring": reference_label,
                "hajek_difference": estimate,
                "spatial_hac_se": hac["standard_error"],
                "hac_n_units": hac["n_units"],
                "hac_n_pairs_used": hac["n_pairs_used"],
                "hac_truncated": hac["truncated"],
                "hac_cutoff_km": cutoff_km,
                "hac_kernel": kernel,
            }
        )
    return pd.DataFrame(rows)


def triangular_kernel_smooth(x: np.ndarray, y: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Nadaraya-Watson smoother with a triangular kernel, for distance-profile figures."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.maximum(1.0 - np.abs(grid[:, None] - x[None, :]) / bandwidth, 0.0)
    denom = weights.sum(axis=1)
    return np.divide(weights @ y, denom, out=np.full_like(grid, np.nan, dtype=float), where=denom > 0)


def triangular_kernel_smooth_se(x: np.ndarray, se: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Standard error of triangular_kernel_smooth, treating each input point's SE as independent."""
    x = np.asarray(x, dtype=float)
    se = np.asarray(se, dtype=float)
    weights = np.maximum(1.0 - np.abs(grid[:, None] - x[None, :]) / bandwidth, 0.0)
    denom = weights.sum(axis=1)
    variance = (weights**2) @ (se**2)
    return np.divide(np.sqrt(variance), denom, out=np.full_like(grid, np.nan, dtype=float), where=denom > 0)
