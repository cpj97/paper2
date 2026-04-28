from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scipy is required for notebook 03 spatial neighbor construction. "
        "Run notebook 00 cell 2 (`%pip install -r ../requirements.txt`) "
        "and restart or reselect the notebook kernel before rerunning."
    ) from exc


SOURCE_COL = "source_cell_id"
NEIGHBOR_COL = "neighbor_cell_id"


@dataclass(frozen=True)
class InverseDistanceSpec:
    support_name: str
    power: float = 1.0
    min_distance_m: float = 1.0
    row_standardize: bool = True


def _support_slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def inverse_distance_support_name(spec: InverseDistanceSpec) -> str:
    return f"invdist_{spec.support_name}_p{_support_slug(spec.power)}"


def build_grid_centroids(
    grid_gdf: gpd.GeoDataFrame,
    grid_cell_id_col: str = "cell_id",
    area_crs: str = "EPSG:6933",
) -> gpd.GeoDataFrame:
    grid = grid_gdf[[grid_cell_id_col, "geometry"]].copy()
    grid = grid[grid.geometry.notna()].drop_duplicates(subset=[grid_cell_id_col]).copy()
    if grid.empty:
        return gpd.GeoDataFrame(
            columns=[grid_cell_id_col, "centroid_x_m", "centroid_y_m", "centroid_lon", "centroid_lat", "geometry"],
            geometry="geometry",
            crs=area_crs,
        )
    if grid.crs is None:
        raise ValueError("Grid geometry is missing a CRS.")

    grid[grid_cell_id_col] = grid[grid_cell_id_col].astype("string")
    grid_area = grid.to_crs(area_crs).reset_index(drop=True)
    centroids_geom = grid_area.geometry.centroid
    centroids = gpd.GeoDataFrame(
        {
            grid_cell_id_col: grid_area[grid_cell_id_col].astype("string"),
            "centroid_x_m": centroids_geom.x,
            "centroid_y_m": centroids_geom.y,
        },
        geometry=centroids_geom,
        crs=area_crs,
    )
    centroids_ll = centroids.to_crs("EPSG:4326")
    centroids["centroid_lon"] = centroids_ll.geometry.x
    centroids["centroid_lat"] = centroids_ll.geometry.y
    return centroids[[grid_cell_id_col, "centroid_x_m", "centroid_y_m", "centroid_lon", "centroid_lat", "geometry"]]


def _coords_from_centroids(centroid_gdf: gpd.GeoDataFrame) -> np.ndarray:
    if not {"centroid_x_m", "centroid_y_m"}.issubset(centroid_gdf.columns):
        raise ValueError("Centroid GeoDataFrame must include 'centroid_x_m' and 'centroid_y_m'.")
    return np.column_stack(
        [
            centroid_gdf["centroid_x_m"].to_numpy(dtype=float),
            centroid_gdf["centroid_y_m"].to_numpy(dtype=float),
        ]
    )


def _cell_ids(centroid_gdf: gpd.GeoDataFrame, grid_cell_id_col: str) -> np.ndarray:
    return centroid_gdf[grid_cell_id_col].astype("string").to_numpy()


def _empty_edges(
    support_name: str,
    support_type: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            SOURCE_COL,
            NEIGHBOR_COL,
            "distance_m",
            "binary_weight",
            "support_name",
            "support_type",
            "neighbor_rank",
        ]
    )


def _query_pairs(tree: cKDTree, radius_m: float) -> np.ndarray:
    try:
        pairs = tree.query_pairs(radius_m, output_type="ndarray")
    except TypeError:
        pairs = np.array(list(tree.query_pairs(radius_m)), dtype=int)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, 2)
    return pairs


def build_radius_neighbor_edges(
    centroid_gdf: gpd.GeoDataFrame,
    radius_m: float,
    grid_cell_id_col: str = "cell_id",
    support_name: str | None = None,
) -> pd.DataFrame:
    if centroid_gdf.empty:
        return _empty_edges(support_name or f"radius_{int(radius_m)}m", "radius")

    coords = _coords_from_centroids(centroid_gdf)
    ids = _cell_ids(centroid_gdf, grid_cell_id_col)
    tree = cKDTree(coords)
    pairs = _query_pairs(tree, radius_m)
    if pairs.size == 0:
        return _empty_edges(support_name or f"radius_{int(radius_m)}m", "radius")

    deltas = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    dist = np.sqrt((deltas**2).sum(axis=1))

    src_idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
    nbr_idx = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dist_full = np.concatenate([dist, dist])

    edges = pd.DataFrame(
        {
            SOURCE_COL: ids[src_idx],
            NEIGHBOR_COL: ids[nbr_idx],
            "distance_m": dist_full,
            "binary_weight": 1.0,
            "support_name": support_name or f"radius_{int(radius_m)}m",
            "support_type": "radius",
            "neighbor_rank": pd.Series([pd.NA] * len(src_idx), dtype="Int64"),
        }
    )
    return edges.sort_values([SOURCE_COL, "distance_m", NEIGHBOR_COL]).reset_index(drop=True)


def build_knn_neighbor_edges(
    centroid_gdf: gpd.GeoDataFrame,
    k: int,
    grid_cell_id_col: str = "cell_id",
    support_name: str | None = None,
) -> pd.DataFrame:
    if centroid_gdf.empty:
        return _empty_edges(support_name or f"knn_k{k}", "knn")

    coords = _coords_from_centroids(centroid_gdf)
    ids = _cell_ids(centroid_gdf, grid_cell_id_col)
    tree = cKDTree(coords)
    max_k = min(int(k), max(len(coords) - 1, 0))
    if max_k <= 0:
        return _empty_edges(support_name or f"knn_k{k}", "knn")

    dist, idx = tree.query(coords, k=max_k + 1)
    if max_k == 1:
        dist = dist.reshape(-1, 2)
        idx = idx.reshape(-1, 2)

    dist = dist[:, 1 : max_k + 1]
    idx = idx[:, 1 : max_k + 1]

    source_ids = np.repeat(ids, max_k)
    neighbor_ids = ids[idx.reshape(-1)]
    neighbor_ranks = np.tile(np.arange(1, max_k + 1), len(ids))

    edges = pd.DataFrame(
        {
            SOURCE_COL: source_ids,
            NEIGHBOR_COL: neighbor_ids,
            "distance_m": dist.reshape(-1),
            "binary_weight": 1.0,
            "support_name": support_name or f"knn_k{max_k}",
            "support_type": "knn",
            "neighbor_rank": pd.Series(neighbor_ranks, dtype="Int64"),
        }
    )
    return edges.sort_values([SOURCE_COL, "neighbor_rank", NEIGHBOR_COL]).reset_index(drop=True)


def add_inverse_distance_weights(
    edges_df: pd.DataFrame,
    power: float = 1.0,
    min_distance_m: float = 1.0,
    row_standardize: bool = True,
    support_name: str | None = None,
) -> pd.DataFrame:
    edges = edges_df.copy()
    if edges.empty:
        edges["raw_weight"] = pd.Series(dtype=float)
        edges["weight"] = pd.Series(dtype=float)
        edges["support_name"] = support_name or "inverse_distance"
        edges["support_type"] = "inverse_distance"
        return edges

    safe_distance = edges["distance_m"].clip(lower=float(min_distance_m))
    edges["raw_weight"] = np.power(safe_distance, -float(power))
    if row_standardize:
        denom = edges.groupby(SOURCE_COL)["raw_weight"].transform("sum")
        edges["weight"] = np.where(denom > 0, edges["raw_weight"] / denom, 0.0)
    else:
        edges["weight"] = edges["raw_weight"]
    edges["support_name"] = support_name or edges.get("support_name", "inverse_distance")
    edges["support_type"] = "inverse_distance"
    return edges


def build_ring_neighbor_edges(
    centroid_gdf: gpd.GeoDataFrame,
    inner_radius_m: float,
    outer_radius_m: float,
    grid_cell_id_col: str = "cell_id",
    support_name: str | None = None,
) -> pd.DataFrame:
    if outer_radius_m <= inner_radius_m:
        raise ValueError("outer_radius_m must be larger than inner_radius_m.")
    outer_edges = build_radius_neighbor_edges(
        centroid_gdf,
        radius_m=outer_radius_m,
        grid_cell_id_col=grid_cell_id_col,
        support_name=support_name or f"ring_{int(inner_radius_m)}_{int(outer_radius_m)}m",
    )
    if inner_radius_m <= 0 or outer_edges.empty:
        outer_edges["support_type"] = "ring"
        return outer_edges

    ring_edges = outer_edges[outer_edges["distance_m"] > float(inner_radius_m)].copy()
    ring_edges["support_name"] = support_name or f"ring_{int(inner_radius_m)}_{int(outer_radius_m)}m"
    ring_edges["support_type"] = "ring"
    return ring_edges.reset_index(drop=True)


def summarize_weight_structure(
    edges_df: pd.DataFrame,
    all_cell_ids: pd.Series | np.ndarray,
    support_name: str | None = None,
) -> dict:
    cell_ids = pd.Series(all_cell_ids, dtype="string")
    if edges_df.empty:
        return {
            "support_name": support_name or None,
            "support_type": None,
            "n_directed_edges": 0,
            "n_undirected_pairs": 0,
            "mean_neighbors": 0.0,
            "median_neighbors": 0.0,
            "max_neighbors": 0,
            "share_cells_with_neighbors": 0.0,
            "distance_km_min": None,
            "distance_km_p50": None,
            "distance_km_mean": None,
            "distance_km_p95": None,
            "distance_km_max": None,
        }

    counts = (
        edges_df.groupby(SOURCE_COL)[NEIGHBOR_COL]
        .size()
        .reindex(cell_ids, fill_value=0)
        .astype(int)
    )
    distance_km = edges_df["distance_m"] / 1000.0
    return {
        "support_name": support_name or str(edges_df["support_name"].iloc[0]),
        "support_type": str(edges_df["support_type"].iloc[0]),
        "n_directed_edges": int(edges_df.shape[0]),
        "n_undirected_pairs": int(edges_df.shape[0] // 2) if str(edges_df["support_type"].iloc[0]) in {"radius", "ring"} else None,
        "mean_neighbors": float(counts.mean()),
        "median_neighbors": float(counts.median()),
        "max_neighbors": int(counts.max()),
        "share_cells_with_neighbors": float((counts > 0).mean()),
        "distance_km_min": float(distance_km.min()) if not distance_km.empty else None,
        "distance_km_p50": float(distance_km.quantile(0.5)) if not distance_km.empty else None,
        "distance_km_mean": float(distance_km.mean()) if not distance_km.empty else None,
        "distance_km_p95": float(distance_km.quantile(0.95)) if not distance_km.empty else None,
        "distance_km_max": float(distance_km.max()) if not distance_km.empty else None,
    }


def build_exposure_mapping_catalog(
    radius_values_km: list[float],
    k_values: list[int],
    inverse_distance_specs: list[InverseDistanceSpec],
    ring_bounds_km: list[tuple[float, float]],
    treated_indicator_template: str = "treated_it_{treatment_key}",
) -> pd.DataFrame:
    rows: list[dict] = []

    for radius_km in radius_values_km:
        support_name = f"radius_{int(radius_km)}km"
        rows.append(
            {
                "mapping_name": f"any_treated_neighbor_{support_name}",
                "mapping_family": "any_treated_neighbor_within_radius",
                "support_name": support_name,
                "aggregation": "max",
                "weight_column": "binary_weight",
                "treated_indicator_template": treated_indicator_template,
                "output_scale": "binary",
                "description": f"Equals one if any neighbor within {radius_km:g} km is treated in year t.",
            }
        )
        rows.append(
            {
                "mapping_name": f"share_treated_neighbors_{support_name}",
                "mapping_family": "share_treated_neighbors",
                "support_name": support_name,
                "aggregation": "mean",
                "weight_column": "binary_weight",
                "treated_indicator_template": treated_indicator_template,
                "output_scale": "[0,1]",
                "description": f"Share of neighbors within {radius_km:g} km that are treated in year t.",
            }
        )

    for k in k_values:
        support_name = f"knn_k{k:02d}"
        rows.append(
            {
                "mapping_name": f"share_treated_neighbors_{support_name}",
                "mapping_family": "share_treated_neighbors",
                "support_name": support_name,
                "aggregation": "mean",
                "weight_column": "binary_weight",
                "treated_indicator_template": treated_indicator_template,
                "output_scale": "[0,1]",
                "description": f"Share of the {k} nearest neighbors that are treated in year t.",
            }
        )

    for spec in inverse_distance_specs:
        invdist_name = inverse_distance_support_name(spec)
        rows.append(
            {
                "mapping_name": f"weighted_exposure_{invdist_name}",
                "mapping_family": "inverse_distance_weighted_exposure",
                "support_name": invdist_name,
                "aggregation": "weighted_mean" if spec.row_standardize else "weighted_sum",
                "weight_column": "weight",
                "treated_indicator_template": treated_indicator_template,
                "output_scale": "[0,1]" if spec.row_standardize else "nonnegative",
                "description": (
                    f"Inverse-distance exposure over {spec.support_name} with power={spec.power:g}"
                    + (" and row-standardized weights." if spec.row_standardize else ".")
                ),
            }
        )

    for inner_km, outer_km in ring_bounds_km:
        support_name = f"ring_{int(inner_km)}_{int(outer_km)}km"
        rows.append(
            {
                "mapping_name": f"ring_share_treated_{support_name}",
                "mapping_family": "ring_based_exposure",
                "support_name": support_name,
                "aggregation": "mean",
                "weight_column": "binary_weight",
                "treated_indicator_template": treated_indicator_template,
                "output_scale": "[0,1]",
                "description": f"Share of treated neighbors with distance in ({inner_km:g}, {outer_km:g}] km in year t.",
            }
        )

    return pd.DataFrame(rows)


def compute_neighbor_exposure(
    panel_status_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cell_id_col: str,
    year_col: str,
    value_col: str,
    aggregation: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    if panel_status_df.empty or edges_df.empty:
        return pd.DataFrame(columns=[cell_id_col, year_col, "exposure"])

    status = panel_status_df[[cell_id_col, year_col, value_col]].copy()
    status[cell_id_col] = status[cell_id_col].astype("string")
    status[year_col] = status[year_col]
    status[value_col] = pd.to_numeric(status[value_col], errors="coerce").fillna(0.0)

    merged = edges_df[[SOURCE_COL, NEIGHBOR_COL] + ([weight_col] if weight_col else [])].merge(
        status.rename(columns={cell_id_col: NEIGHBOR_COL}),
        on=NEIGHBOR_COL,
        how="left",
    )
    merged[value_col] = merged[value_col].fillna(0.0)

    group_cols = [SOURCE_COL, year_col]
    if aggregation == "max":
        exposure = merged.groupby(group_cols, as_index=False)[value_col].max()
    elif aggregation == "mean":
        exposure = merged.groupby(group_cols, as_index=False)[value_col].mean()
    elif aggregation == "weighted_mean":
        if weight_col is None or weight_col not in merged.columns:
            raise ValueError("weighted_mean requires a valid weight_col.")
        merged["_weighted_value"] = merged[value_col] * merged[weight_col]
        exposure = merged.groupby(group_cols, as_index=False)["_weighted_value"].sum().rename(columns={"_weighted_value": "exposure"})
        exposure = exposure.rename(columns={SOURCE_COL: cell_id_col})
        return exposure[[cell_id_col, year_col, "exposure"]]
    elif aggregation == "weighted_sum":
        if weight_col is None or weight_col not in merged.columns:
            raise ValueError("weighted_sum requires a valid weight_col.")
        merged["_weighted_value"] = merged[value_col] * merged[weight_col]
        exposure = merged.groupby(group_cols, as_index=False)["_weighted_value"].sum().rename(columns={"_weighted_value": "exposure"})
        exposure = exposure.rename(columns={SOURCE_COL: cell_id_col})
        return exposure[[cell_id_col, year_col, "exposure"]]
    else:
        raise ValueError("aggregation must be one of {'max', 'mean', 'weighted_mean', 'weighted_sum'}.")

    exposure = exposure.rename(columns={SOURCE_COL: cell_id_col, value_col: "exposure"})
    return exposure[[cell_id_col, year_col, "exposure"]]
