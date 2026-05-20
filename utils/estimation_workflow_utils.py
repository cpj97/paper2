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
) -> tuple[float, pd.DataFrame]:
    """No-covariate 2x2 DID influence contributions for one ATT(g,t).

    This is intended as the HAC inference layer for the no-covariate
    specification. The point estimate can be compared with package output.
    """
    use_years = [int(base_year), int(time_year)]
    wide = (
        panel.loc[panel[year_col].isin(use_years), [cell_col, year_col, outcome_col, group_col]]
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
) -> pd.DataFrame:
    rows = []
    for row in effects.itertuples(index=False):
        group_year = int(getattr(row, "group"))
        time_year = int(getattr(row, "time"))
        base_year = group_year - 1
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
            }
        )
    return pd.DataFrame(rows)


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
