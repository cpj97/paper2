"""Protected-area treatment dating from WDPA yearly snapshots.

Shared by notebook 02 (which writes the project's treatment lookup) and notebook 11 (which
compares dating rules). Three rules are computed from one cell x polygon overlay:

* snapshot (original nb02 rule): first snapshot year covering the cell; the polygon's
  STATUS_YR only if the cell is already covered by the earliest snapshot.
* statusyr_first: min(STATUS_YR, snapshot year) of the polygons in the cell's first covering
  snapshot.
* statusyr_expansions (definition C): as statusyr_first, but a cell added to a park that already
  appeared in an earlier snapshot is an expansion and gets the expansion's own year: the legal
  year from `expansion_legal_years` when known, otherwise the first snapshot showing it. The
  park's earlier area keeps its earlier year. Expansions before the earliest snapshot cannot be
  detected (no earlier boundary to compare against).

Coverage uses positive-area overlap, which is nb02's `any` overlay rule; the overlay reproduces
nb02's original dates exactly for every protected cell.
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def wdpa_site_ids(wdpa: pd.DataFrame) -> pd.Series:
    """Stable WDPA site id across snapshots: WDPAID (to 2024) or SITE_ID (2025 schema)."""
    wdpaid = pd.to_numeric(wdpa["WDPAID"], errors="coerce") if "WDPAID" in wdpa.columns else pd.Series(np.nan, index=wdpa.index)
    site_id = pd.to_numeric(wdpa["SITE_ID"], errors="coerce") if "SITE_ID" in wdpa.columns else pd.Series(np.nan, index=wdpa.index)
    return wdpaid.fillna(site_id).astype("Int64")


def prepare_wdpa_dating_frame(
    wdpa: gpd.GeoDataFrame,
    *,
    status_col: str = "STATUS_YR",
    snapshot_col: str = "wdpa_snapshot_year",
    status_bounds: tuple[int, int] = (1900, 2025),
    max_snapshot_year: int | None = None,
) -> gpd.GeoDataFrame:
    """Add site_id, status (valid STATUS_YR, not after the snapshot) and treat_year."""
    w = wdpa[wdpa.geometry.notna()].copy()
    w[snapshot_col] = pd.to_numeric(w[snapshot_col], errors="coerce")
    w = w[w[snapshot_col].notna()].copy()
    w[snapshot_col] = w[snapshot_col].astype(int)
    if max_snapshot_year is not None:
        w = w[w[snapshot_col] <= int(max_snapshot_year)].copy()
    w["site_id"] = wdpa_site_ids(w)
    status = pd.to_numeric(w[status_col], errors="coerce").round()
    status = status.where(status.between(*status_bounds))
    w["status"] = status.where(status <= w[snapshot_col])
    w["treat_year"] = w["status"].fillna(w[snapshot_col]).astype(int)
    return w


def build_overlap_pairs(
    wdpa: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    *,
    cell_col: str = "cell_id",
    snapshot_col: str = "wdpa_snapshot_year",
    area_crs: str = "EPSG:6933",
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per (cell, polygon) with positive-area overlap.

    `wdpa` must come from prepare_wdpa_dating_frame. Identical (site, date, geometry) rows
    repeated across snapshots are tested once, keeping their earliest snapshot. Cells strictly
    inside a polygon are accepted with a prepared-geometry test; boundary cells are accepted only
    if their interior overlaps the polygon (relate pattern T********), i.e. positive overlap
    area, without building intersection geometries.
    """
    t0 = time.time()
    g = grid[[cell_col, "geometry"]].copy()
    if g.crs is None or g.crs.to_string() != area_crs:
        g = g.to_crs(area_crs)
    g = g.reset_index(drop=True)

    w = wdpa[["site_id", "status", "treat_year", snapshot_col, "geometry"]].copy()
    w["_wkb"] = shapely.to_wkb(w.geometry.values)
    w = w.sort_values(snapshot_col).drop_duplicates(subset=["site_id", "treat_year", "_wkb"]).drop(columns="_wkb")
    w["geometry"] = w.geometry.make_valid()
    w = w.to_crs(area_crs).reset_index(drop=True)
    if verbose:
        print(f"overlay: {len(g):,} cells, {len(w):,} unique polygons  [{time.time() - t0:.0f}s]")

    polys = np.asarray(w.geometry.values, dtype=object)
    cell_geoms = np.asarray(g.geometry.values, dtype=object)
    pi, ci = shapely.STRtree(cell_geoms).query(polys, predicate="intersects")
    order = np.argsort(pi, kind="stable")
    pi, ci = pi[order], ci[order]
    bounds = np.searchsorted(pi, np.arange(len(polys) + 1))
    ok = np.zeros(len(pi), dtype=bool)
    for k in range(len(polys)):
        a, b = bounds[k], bounds[k + 1]
        if a == b:
            continue
        poly = polys[k]
        shapely.prepare(poly)
        sub = cell_geoms[ci[a:b]]
        res = shapely.contains_properly(poly, sub)
        edge = ~res
        if edge.any():
            res[edge] = shapely.relate_pattern(poly, sub[edge], "T********")
        ok[a:b] = res
        shapely.destroy_prepared(poly)
    if verbose:
        print(f"overlay: {len(pi):,} candidate pairs, {int(ok.sum()):,} overlapping  [{time.time() - t0:.0f}s]")
    return pd.DataFrame({
        cell_col: g[cell_col].to_numpy()[ci[ok]],
        "site_id": w["site_id"].to_numpy()[pi[ok]],
        "snap": w[snapshot_col].to_numpy()[pi[ok]].astype(int),
        "status": w["status"].to_numpy()[pi[ok]],
        "treat": w["treat_year"].to_numpy()[pi[ok]],
    })


def date_cells_from_pairs(
    pairs: pd.DataFrame,
    site_first_snapshot: pd.Series,
    *,
    cell_col: str = "cell_id",
    earliest_snapshot: int = 2017,
    expansion_legal_years: dict | None = None,
) -> pd.DataFrame:
    """Apply the three dating rules to the overlay. One row per protected cell."""
    legal_years = expansion_legal_years or {}
    first = pairs.groupby(cell_col)["snap"].transform("min")
    fp = pairs[pairs["snap"] == first].copy()
    fp["site_first_snap"] = fp["site_id"].map(site_first_snapshot)
    fp["is_expansion"] = (fp["snap"] > int(earliest_snapshot)) & (fp["site_first_snap"] < fp["snap"])
    legal = fp["site_id"].map(lambda s: legal_years.get(int(s)) if pd.notna(s) else None).astype("float")
    fp["year_c"] = np.where(fp["is_expansion"], legal.fillna(fp["snap"]), fp["treat"]).astype(int)
    fp["c_source"] = np.where(~fp["is_expansion"], "STATUS_YR rule",
                      np.where(legal.notna(), "expansion: legal year (manual)", "expansion: first snapshot showing it"))

    out = fp.groupby(cell_col).agg(
        first_snap=("snap", "min"), status_first=("status", "min"),
        year_statusyr_first=("treat", "min"), year_statusyr_expansions=("year_c", "min"))
    out["year_snapshot"] = np.where((out["first_snap"] == int(earliest_snapshot)) & out["status_first"].notna(),
                                    out["status_first"], out["first_snap"]).astype(int)
    out["year_statusyr_min"] = pairs.groupby(cell_col)["treat"].min().reindex(out.index)
    src = fp.sort_values("year_c").drop_duplicates(cell_col).set_index(cell_col)[["c_source", "site_id"]]
    out = out.join(src.rename(columns={"site_id": "c_site_id"}))
    for col in ["year_statusyr_first", "year_statusyr_expansions", "year_snapshot", "year_statusyr_min"]:
        out[col] = out[col].astype(int)
    return out.drop(columns=["status_first"]).reset_index()
