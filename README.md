# Geospatial Causal Workflow Starter

This repository contains notebooks for the first four steps of the workflow:

- `notebooks/00_project_setup.ipynb`
- `notebooks/01_build_grid_panel.ipynb`
- `notebooks/02_define_treatment_timing.ipynb`
- `notebooks/03_build_spatial_structure.ipynb`

The workflow does not depend on a local `src/` package, but notebook 03 imports lightweight helper modules from `utils/`. Run notebook 00 first so the required environment, including `scipy`, is available in the active kernel.

## Recommended environment

- Python 3.13+
- Exact package versions are recorded in:
  - `requirements.txt`
  - `configs/environment_manifest.json` after running notebook 00

## Project structure

```text
paper2/
|-- notebooks/
|-- utils/
|-- configs/
|-- data/
|   |-- raw/
|   |-- intermediate/
|   `-- processed/
|-- outputs/
|   |-- figures/
|   |-- maps/
|   `-- tables/
|-- paper/
`-- logs/
```

## Suggested usage

1. Open the project root in Jupyter or VS Code.
2. Run `notebooks/00_project_setup.ipynb`.
3. If you use the install cell in notebook 00, restart or re-select the kernel after `%pip install -r ../requirements.txt`.
4. Fill in the raw data paths in `notebooks/01_build_grid_panel.ipynb`.
5. Run `notebooks/01_build_grid_panel.ipynb`.
6. Set the treatment source configuration in `notebooks/02_define_treatment_timing.ipynb`.
7. Run `notebooks/02_define_treatment_timing.ipynb`.
8. Run `notebooks/03_build_spatial_structure.ipynb` to build centroids, neighbor structures, candidate spatial weights, and exposure mappings.

The notebooks write out configuration files and intermediate data products that later notebooks can read.
