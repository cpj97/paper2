# Geospatial Causal Workflow Starter (Inline-Notebook Version)

This package contains self-contained notebooks for the first two steps of the workflow:

- `notebooks/00_project_setup.ipynb`
- `notebooks/01_build_grid_panel.ipynb`

These notebooks **do not depend on a local `src/` package**. All helper functions needed for these two steps are defined inside the notebooks to make replication easier.

## Recommended environment

- Python 3.11.8
- Exact package versions are recorded in:
  - `requirements.txt`
  - `configs/environment_manifest.json` (created by Notebook 00)

## Project structure

```text
geospatial_causal_workflow_inline/
├─ notebooks/
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ intermediate/
│  └─ processed/
├─ outputs/
│  ├─ figures/
│  ├─ maps/
│  └─ tables/
└─ logs/
```

## Suggested usage

1. Open the project root in Jupyter or VS Code.
2. Run `00_project_setup.ipynb`.
3. Fill in the raw data paths in `01_build_grid_panel.ipynb`.
4. Run `01_build_grid_panel.ipynb`.

The notebooks also write out configuration files that later notebooks can read.
