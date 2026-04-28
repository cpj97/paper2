from __future__ import annotations

import geopandas as gpd
import pandas as pd


def _empty_lookup(panel_cell_id_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            panel_cell_id_col,
            "first_treat_year",
            "max_overlap_share",
            "total_overlap_area_m2",
            "source_rows",
            "treatment_key",
        ]
    )


def prepare_lookup_grid(
    grid_gdf: gpd.GeoDataFrame,
    grid_cell_id_col: str = "cell_id",
    area_crs: str = "EPSG:6933",
) -> gpd.GeoDataFrame:
    grid = grid_gdf[[grid_cell_id_col, "geometry"]].copy()
    grid = grid[grid.geometry.notna()].drop_duplicates(subset=[grid_cell_id_col]).copy()
    grid[grid_cell_id_col] = grid[grid_cell_id_col].astype("string")
    grid = grid.to_crs(area_crs).reset_index(drop=True)
    grid["cell_area_m2"] = grid.geometry.area
    return grid[[grid_cell_id_col, "cell_area_m2", "geometry"]]


def build_lookup_from_timed_polygons_fast(
    source_gdf: gpd.GeoDataFrame,
    grid_area_gdf: gpd.GeoDataFrame,
    treatment_key: str,
    panel_cell_id_col: str = "cell_id",
    grid_cell_id_col: str = "cell_id",
    polygon_overlay_rule: str = "any",
    min_cell_overlap_share: float = 0.01,
) -> pd.DataFrame:
    empty_lookup = _empty_lookup(panel_cell_id_col)
    if source_gdf is None or source_gdf.empty or grid_area_gdf is None or grid_area_gdf.empty:
        return empty_lookup

    source = source_gdf[["treat_year", "geometry"]].copy()
    try:
        source["geometry"] = source.geometry.make_valid()
    except Exception:
        pass
    source = source[source.geometry.notna()].copy()
    source["treat_year"] = pd.to_numeric(source["treat_year"], errors="coerce").astype("Int64")
    source = source[source["treat_year"].notna()].copy()
    if source.empty:
        return empty_lookup
    if source.crs is None:
        raise ValueError(f"{treatment_key} source geometries are missing a CRS.")

    source = source.to_crs(grid_area_gdf.crs).reset_index(drop=True)
    grid = grid_area_gdf[[grid_cell_id_col, "cell_area_m2", "geometry"]].copy().reset_index(drop=True)

    joined = gpd.sjoin(
        source[["treat_year", "geometry"]],
        grid,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return empty_lookup

    joined = joined.reset_index(drop=True)
    grid_geom = grid.geometry.iloc[joined["index_right"].to_numpy()].reset_index(drop=True)
    clipped = joined.geometry.reset_index(drop=True).intersection(grid_geom)
    overlap = gpd.GeoDataFrame(
        joined[["treat_year", grid_cell_id_col, "cell_area_m2"]].copy(),
        geometry=clipped,
        crs=grid.crs,
    )
    overlap = overlap[overlap.geometry.notna() & ~overlap.geometry.is_empty].copy()
    if overlap.empty:
        return empty_lookup

    overlap["piece_area_m2"] = overlap.geometry.area
    overlap = overlap[overlap["piece_area_m2"] > 0].copy()
    if overlap.empty:
        return empty_lookup

    # Union clipped pieces only within the same cell-year pair. This avoids
    # the national-scale dissolve that was dominating runtime.
    cell_year = overlap.dissolve(by=[grid_cell_id_col, "treat_year"], aggfunc={"cell_area_m2": "first"}).reset_index()
    cell_year["overlap_area_m2"] = cell_year.geometry.area
    cell_year["overlap_share"] = cell_year["overlap_area_m2"] / cell_year["cell_area_m2"]

    if polygon_overlay_rule == "share":
        cell_year = cell_year[cell_year["overlap_share"] >= float(min_cell_overlap_share)].copy()
    elif polygon_overlay_rule == "any":
        cell_year = cell_year[cell_year["overlap_area_m2"] > 0].copy()
    else:
        raise ValueError("polygon_overlay_rule must be either 'any' or 'share'.")

    if cell_year.empty:
        return empty_lookup

    lookup = (
        cell_year.groupby(grid_cell_id_col, as_index=False)
        .agg(
            first_treat_year=("treat_year", "min"),
            max_overlap_share=("overlap_share", "max"),
            total_overlap_area_m2=("overlap_area_m2", "sum"),
            source_rows=("treat_year", "size"),
        )
        .rename(columns={grid_cell_id_col: panel_cell_id_col})
    )
    lookup[panel_cell_id_col] = lookup[panel_cell_id_col].astype("string")
    lookup["first_treat_year"] = lookup["first_treat_year"].astype("Int64")
    lookup["treatment_key"] = treatment_key
    return lookup
