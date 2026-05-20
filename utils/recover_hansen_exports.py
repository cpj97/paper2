"""Recover and assemble Hansen GEE exports after a notebook/kernel interruption.

This script reconstructs the expected Google Drive export filenames from the
Notebook 01 run tag, downloads missing CSV pieces and grid geometry, validates
the pieces, and writes the same raw wide/long parquet outputs that Notebook 01
would create after the Earth Engine tasks finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def build_year_blocks(years: list[int], block_size: int) -> list[list[int]]:
    years = sorted(int(y) for y in years)
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    return [years[i : i + block_size] for i in range(0, len(years), block_size)]


def build_export_specs(
    export_prefix: str,
    years: list[int],
    block_size: int = 5,
    n_chunks: int = 1,
    include_base: bool = True,
) -> list[dict]:
    specs: list[dict] = []
    chunk_ids = list(range(max(1, int(n_chunks))))

    if include_base:
        for chunk_id in chunk_ids:
            basename = f"{export_prefix}_base"
            if n_chunks > 1:
                basename += f"_chunk{chunk_id:02d}"
            specs.append(
                {
                    "kind": "base",
                    "years": [],
                    "chunk_id": chunk_id,
                    "basename": basename,
                    "filename": f"{basename}.csv",
                    "value_cols": ["base_m2"],
                }
            )

    for block in build_year_blocks(years, block_size):
        start_year, end_year = block[0], block[-1]
        value_cols = [f"loss_{y}_m2" for y in block]
        for chunk_id in chunk_ids:
            basename = f"{export_prefix}_loss_{start_year}_{end_year}"
            if n_chunks > 1:
                basename += f"_chunk{chunk_id:02d}"
            specs.append(
                {
                    "kind": "loss",
                    "years": block,
                    "chunk_id": chunk_id,
                    "basename": basename,
                    "filename": f"{basename}.csv",
                    "value_cols": value_cols,
                }
            )
    return specs


def get_drive_service(project_root: Path):
    client_secret_file = project_root / "configs" / "google_drive_client_secret.json"
    token_file = project_root / "configs" / "google_drive_token.json"

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
            creds = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def find_drive_folder_id(service, folder_name: str) -> str:
    q = (
        f"name = '{folder_name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    results = (
        service.files()
        .list(q=q, spaces="drive", fields="files(id, name, modifiedTime)", orderBy="modifiedTime desc")
        .execute()
    )
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(f"Drive folder {folder_name!r} not found.")
    if len(files) > 1:
        print(f"Found {len(files)} Drive folders named {folder_name!r}; using most recent:")
        for item in files[:10]:
            print(" -", item["id"], item["name"], item.get("modifiedTime"))
    return files[0]["id"]


def list_drive_files(service, folder_id: str) -> dict[str, dict]:
    files_by_name: dict[str, dict] = {}
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        for item in response.get("files", []):
            files_by_name[item["name"]] = item
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files_by_name


def download_drive_file(service, file_id: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with local_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  {local_path.name}: {int(status.progress() * 100)}%")


def coordinate_cell_id(df: pd.DataFrame) -> pd.Series:
    if "cell_lon" not in df.columns or "cell_lat" not in df.columns:
        raise ValueError("Coordinate cell-id repair requires cell_lon and cell_lat columns.")
    lon = pd.to_numeric(df["cell_lon"], errors="raise")
    lat = pd.to_numeric(df["cell_lat"], errors="raise")
    return "lon_" + lon.map(lambda x: f"{x:.6f}") + "_lat_" + lat.map(lambda x: f"{x:.6f}")


def validate_csv_pieces(
    csv_paths: list[Path],
    specs: list[dict],
    id_col: str = "cell_id",
    repair_cell_id_from_coordinates: bool = False,
):
    validation_rows = []
    bad_rows = []

    for path, spec in zip(csv_paths, specs):
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV piece: {path}")

        probe = pd.read_csv(path, nrows=5)
        usecols = [id_col]
        if repair_cell_id_from_coordinates:
            usecols += ["cell_lon", "cell_lat"]
        if "chunk_id" in probe.columns:
            usecols.append("chunk_id")

        df_ids = pd.read_csv(path, usecols=usecols)
        if repair_cell_id_from_coordinates:
            df_ids[id_col] = coordinate_cell_id(df_ids)
        df_ids[id_col] = df_ids[id_col].astype(str)

        file_chunk = int(spec["chunk_id"])
        missing_chunk_col = int("chunk_id" not in df_ids.columns)
        chunk_mismatches = 0
        if "chunk_id" in df_ids.columns:
            chunk_mismatches = int((df_ids["chunk_id"] != file_chunk).sum())

        validation_rows.append(
            {
                "file": path.name,
                "kind": spec["kind"],
                "file_chunk": file_chunk,
                "rows": int(len(df_ids)),
                "duplicate_ids_within_file": int(df_ids[id_col].duplicated().sum()),
                "missing_chunk_col": missing_chunk_col,
                "file_chunk_col_mismatches": chunk_mismatches,
            }
        )

        if missing_chunk_col:
            bad = df_ids.copy()
            bad["file"] = path.name
            bad["file_chunk"] = file_chunk
            bad_rows.append(bad)
        elif chunk_mismatches:
            bad = df_ids.loc[df_ids["chunk_id"] != file_chunk].copy()
            bad["file"] = path.name
            bad["file_chunk"] = file_chunk
            bad_rows.append(bad)

    validation_df = pd.DataFrame(validation_rows).sort_values(["kind", "file_chunk", "file"]).reset_index(drop=True)
    bad_df = pd.concat(bad_rows, ignore_index=True) if bad_rows else pd.DataFrame()
    return validation_df, bad_df


def assemble_csv_pieces(
    csv_paths: list[Path],
    specs: list[dict],
    id_col: str = "cell_id",
    repair_cell_id_from_coordinates: bool = False,
):
    block_dfs: dict[tuple[str, ...], list[pd.DataFrame]] = {}
    piece_summaries = []

    for path, spec in zip(csv_paths, specs):
        df_piece = pd.read_csv(path)
        if id_col not in df_piece.columns:
            raise ValueError(f"{id_col!r} not found in {path.name}: {list(df_piece.columns)}")
        if repair_cell_id_from_coordinates:
            df_piece[id_col] = coordinate_cell_id(df_piece)
        df_piece[id_col] = df_piece[id_col].astype(str)

        keep_cols = [id_col]
        keep_cols += [col for col in ["cell_lon", "cell_lat"] if col in df_piece.columns]
        for value_col in spec["value_cols"]:
            if value_col not in df_piece.columns:
                raise ValueError(f"{value_col!r} not found in {path.name}.")
            keep_cols.append(value_col)

        df_piece = df_piece[keep_cols].copy()
        duplicate_ids = int(df_piece[id_col].duplicated().sum())
        if duplicate_ids:
            raise ValueError(f"{path.name} has {duplicate_ids} duplicate {id_col} values.")

        block_key = tuple(spec["value_cols"])
        block_dfs.setdefault(block_key, []).append(df_piece)
        piece_summaries.append(
            {
                "filename": path.name,
                "n_rows": int(df_piece.shape[0]),
                "n_unique_ids": int(df_piece[id_col].nunique()),
                "n_cols": int(df_piece.shape[1]),
                "kind": spec["kind"],
                "chunk_id": int(spec["chunk_id"]),
                "value_cols": list(spec["value_cols"]),
            }
        )

    merged = None
    for block_key, pieces in block_dfs.items():
        df_block = pd.concat(pieces, ignore_index=True, sort=False)
        duplicate_ids = int(df_block[id_col].duplicated().sum())
        if duplicate_ids:
            if not repair_cell_id_from_coordinates:
                examples = df_block.loc[df_block[id_col].duplicated(keep=False), id_col].drop_duplicates().head(10).tolist()
                raise ValueError(f"Duplicate ids after concatenating block {list(block_key)}: {examples}")
            before = len(df_block)
            df_block = df_block.drop_duplicates(id_col, keep="first").copy()
            print(
                "Dropped",
                before - len(df_block),
                "duplicate repaired coordinate cell(s) for block",
                list(block_key),
            )

        if merged is None:
            merged = df_block
        else:
            drop_cols = [c for c in ["cell_lon", "cell_lat"] if c in df_block.columns and c in merged.columns]
            df_block = df_block.drop(columns=drop_cols, errors="ignore")
            overlapping = [c for c in df_block.columns if c != id_col and c in merged.columns]
            if overlapping:
                raise ValueError(f"Unexpected overlapping columns across blocks: {overlapping}")
            merged = merged.merge(df_block, on=id_col, how="outer", validate="one_to_one")

    if merged is None:
        raise ValueError("No CSV pieces were assembled.")
    return merged.sort_values(id_col).reset_index(drop=True), piece_summaries


def melt_wide_to_long(df_wide: pd.DataFrame, id_col: str = "cell_id") -> pd.DataFrame:
    loss_cols = [c for c in df_wide.columns if c.startswith("loss_") and c.endswith("_m2")]
    df_long = df_wide.melt(
        id_vars=[c for c in df_wide.columns if c not in loss_cols],
        value_vars=loss_cols,
        var_name="loss_band",
        value_name="loss_m2",
    )
    df_long["year"] = df_long["loss_band"].str.extract(r"loss_(\d{4})_m2").astype(int)
    return df_long.drop(columns=["loss_band"])


def normalize_grid_geometry_crs(grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return grid geometry in EPSG:4326 even when EE GeoJSON stores EPSG:3857 coords."""
    if grid_gdf.empty:
        return grid_gdf
    bounds = grid_gdf.total_bounds
    looks_projected = bool(max(abs(bounds[0]), abs(bounds[2])) > 180 or max(abs(bounds[1]), abs(bounds[3])) > 90)
    declared_grid_crs = None
    if "grid_crs" in grid_gdf.columns and grid_gdf["grid_crs"].notna().any():
        declared_grid_crs = str(grid_gdf["grid_crs"].dropna().iloc[0])
    if looks_projected or declared_grid_crs == "EPSG:3857":
        grid_gdf = grid_gdf.set_crs("EPSG:3857", allow_override=True).to_crs("EPSG:4326")
    elif grid_gdf.crs is None:
        grid_gdf = grid_gdf.set_crs("EPSG:4326")
    else:
        grid_gdf = grid_gdf.to_crs("EPSG:4326")
    return grid_gdf


def load_project_config(project_root: Path) -> dict:
    path = project_root / "configs" / "project_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True, help="Run tag, e.g. 20260513_133313.")
    parser.add_argument("--grid-tag", default="1km")
    parser.add_argument("--n-grid-chunks", type=int, default=16)
    parser.add_argument("--year-start", type=int, default=2001)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--year-block-size", type=int, default=5)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--download", action="store_true", help="Download missing pieces from Google Drive.")
    parser.add_argument("--assemble", action="store_true", help="Validate and assemble local pieces.")
    parser.add_argument(
        "--repair-cell-id-from-coordinates",
        action="store_true",
        help=(
            "Replace exported cell_id with a stable lon/lat-derived id. "
            "Use this for the interrupted 1km export where projected IDs were rounded too coarsely."
        ),
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_project_config(project_root)
    raw_dir = Path(config.get("data_dirs", {}).get("raw", project_root / "data" / "raw"))
    intermediate_dir = Path(config.get("data_dirs", {}).get("intermediate", project_root / "data" / "intermediate"))
    table_dir = Path(config.get("output_dirs", {}).get("tables", project_root / "outputs" / "tables"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    export_prefix = f"colombia_hansen_grid_{args.grid_tag}_{args.run_tag}"
    export_folder = f"gee_exports_{args.run_tag}"
    local_export_dir = raw_dir / export_folder
    grid_file_name = f"{export_prefix}_geometry.geojson"
    assembled_name = f"{export_prefix}_wide_assembled.csv.gz"

    years = list(range(args.year_start, args.year_end + 1))
    specs = build_export_specs(
        export_prefix=export_prefix,
        years=years,
        block_size=args.year_block_size,
        n_chunks=args.n_grid_chunks,
        include_base=True,
    )
    expected_csvs = [local_export_dir / spec["filename"] for spec in specs]
    grid_geo_path = local_export_dir / grid_file_name

    print("Run tag:", args.run_tag)
    print("Export folder:", export_folder)
    print("Export prefix:", export_prefix)
    print("Expected CSV pieces:", len(expected_csvs))
    print("Local export dir:", local_export_dir)

    if args.download:
        service = get_drive_service(project_root)
        folder_id = find_drive_folder_id(service, export_folder)
        files_by_name = list_drive_files(service, folder_id)
        expected_names = [spec["filename"] for spec in specs] + [grid_file_name]
        missing_on_drive = [name for name in expected_names if name not in files_by_name]
        if missing_on_drive:
            raise FileNotFoundError(
                "Missing expected files in Drive folder:\n - "
                + "\n - ".join(missing_on_drive[:30])
                + ("\n ..." if len(missing_on_drive) > 30 else "")
            )

        for local_path in expected_csvs + [grid_geo_path]:
            if local_path.exists() and local_path.stat().st_size > 0:
                print("Already local:", local_path.name)
                continue
            print("Downloading:", local_path.name)
            download_drive_file(service, files_by_name[local_path.name]["id"], local_path)

    local_missing = [path for path in expected_csvs + [grid_geo_path] if not path.exists() or path.stat().st_size == 0]
    if local_missing:
        print("Missing local files:", len(local_missing))
        for path in local_missing[:30]:
            print(" -", path)
        if len(local_missing) > 30:
            print(" ...")
        if args.assemble:
            raise FileNotFoundError("Cannot assemble until all expected files are local.")
        return

    if not args.assemble:
        print("All expected files are local. Re-run with --assemble to build parquet outputs.")
        return

    validation_df, bad_rows_df = validate_csv_pieces(
        expected_csvs,
        specs,
        repair_cell_id_from_coordinates=args.repair_cell_id_from_coordinates,
    )
    validation_path = table_dir / "01_csv_piece_validation_summary.csv"
    validation_df.to_csv(validation_path, index=False)
    print("Saved validation summary:", validation_path)

    problem_files = validation_df[
        (validation_df["duplicate_ids_within_file"] > 0)
        | (validation_df["missing_chunk_col"] > 0)
        | (validation_df["file_chunk_col_mismatches"] > 0)
    ]
    if not bad_rows_df.empty:
        bad_rows_path = table_dir / "01_csv_piece_validation_bad_rows.csv"
        bad_rows_df.to_csv(bad_rows_path, index=False)
        print("Saved bad-row details:", bad_rows_path)
    if not problem_files.empty:
        raise ValueError("CSV piece validation failed:\n" + problem_files.to_string(index=False))

    df_wide, piece_summaries = assemble_csv_pieces(
        expected_csvs,
        specs,
        repair_cell_id_from_coordinates=args.repair_cell_id_from_coordinates,
    )
    assembled_path = intermediate_dir / assembled_name
    df_wide.to_csv(assembled_path, index=False, compression="gzip")
    print("Saved assembled wide CSV:", assembled_path)

    summary_path = table_dir / "01_csv_piece_summary.json"
    summary_path.write_text(json.dumps(piece_summaries, indent=2))
    print("Saved piece summary:", summary_path)

    numeric_cols = [c for c in df_wide.columns if c == "base_m2" or (c.startswith("loss_") and c.endswith("_m2"))]
    for col in numeric_cols:
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce").fillna(0)
    keep_cols = [c for c in ["cell_id", "cell_lon", "cell_lat", "base_m2"] if c in df_wide.columns]
    keep_cols += [c for c in df_wide.columns if c.startswith("loss_") and c.endswith("_m2")]
    df_wide = df_wide[keep_cols].copy()
    df_long = melt_wide_to_long(df_wide)

    wide_parquet_path = intermediate_dir / f"panel_raw_wide_{args.grid_tag}.parquet"
    long_parquet_path = intermediate_dir / f"panel_raw_long_{args.grid_tag}.parquet"
    grid_copy_path = intermediate_dir / f"grid_geometry_{args.grid_tag}.geojson"

    df_wide.to_parquet(wide_parquet_path, index=False)
    df_long.to_parquet(long_parquet_path, index=False)
    grid_gdf = normalize_grid_geometry_crs(gpd.read_file(grid_geo_path))
    if args.repair_cell_id_from_coordinates:
        grid_gdf["cell_id"] = coordinate_cell_id(grid_gdf)
        before_grid = len(grid_gdf)
        grid_gdf = grid_gdf.drop_duplicates("cell_id", keep="first").copy()
        print("Dropped duplicate repaired grid geometries:", before_grid - len(grid_gdf))
    grid_gdf.to_file(grid_copy_path, driver="GeoJSON")

    print("Saved:")
    print(" -", wide_parquet_path)
    print(" -", long_parquet_path)
    print(" -", grid_copy_path)
    print("Wide shape:", df_wide.shape)
    print("Long shape:", df_long.shape)
    print("Grid cells:", len(grid_gdf))


if __name__ == "__main__":
    main()
