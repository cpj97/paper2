"""Run Callaway & Sant'Anna's R `did` package from Python (utils/r/did_runner.R).

The R package is the reference estimator. Python writes the outcome panel and cell table once,
R estimates ATT(g,t) and the simple / dynamic aggregates for each run, and exports the
aggregates' influence functions together with the unit ids in the influence functions' row
order (with faster_mode the order is not sorted by id). Python then applies the spatial-HAC
layer (fast_did_utils.ConleyLattice) to psi = IF / n, where n is the number of units in the run;
with a diagonal kernel this reproduces R's analytic SE, sqrt(mean(IF^2) / n).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

RUNNER = Path(__file__).resolve().parent / "r" / "did_runner.R"


def find_rscript() -> str:
    candidates = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if not candidates:
        raise FileNotFoundError("Rscript.exe not found under C:/Program Files/R")
    return str(candidates[0])


def write_r_inputs(work_dir: Path, *, uid: np.ndarray, years: np.ndarray, Y: np.ndarray, g_did: np.ndarray,
                   keep: np.ndarray, control_flags: dict, covariates: pd.DataFrame,
                   treat_flags: dict | None = None, covariates_by_year: dict | None = None) -> tuple[Path, Path]:
    """long.csv (uid, year, y, time-varying covariates) and cells.csv (uid, g, ctrl_*, trt_*, time-invariant covariates)
    for the cells in `keep`.

    treat_flags (optional) {name: bool mask}: a run with treat_flag = "trt_<name>" treats only the flagged cells with g > 0.
    covariates_by_year (optional) {name: (n_cells x n_years) array}: time-varying covariates; R `did` uses each one's value
    in the comparison's base period (see utils/covariate_windows.py).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = np.flatnonzero(keep)
    long = pa.table({
        "uid": np.repeat(uid[rows], len(years)).astype(np.int32),
        "year": np.tile(years, len(rows)).astype(np.int32),
        "y": Y[rows].ravel().astype(np.float64),
        **{name: np.asarray(m, dtype=np.float64)[rows].ravel() for name, m in (covariates_by_year or {}).items()},
    })
    long_path = work_dir / "long.csv"
    pacsv.write_csv(long, long_path)
    cells = {"uid": uid[rows].astype(np.int32), "g": g_did[rows].astype(np.int32)}
    for name, mask in control_flags.items():
        cells[f"ctrl_{name}"] = mask[rows].astype(np.int8)
    for name, mask in (treat_flags or {}).items():
        cells[f"trt_{name}"] = mask[rows].astype(np.int8)
    for col in covariates.columns:
        cells[col] = covariates[col].to_numpy(dtype=float)[rows]
    cells_path = work_dir / "cells.csv"
    pacsv.write_csv(pa.table(cells), cells_path)
    return long_path, cells_path


def run_r(runs: pd.DataFrame, work_dir: Path, long_path: Path, cells_path: Path, out_dir: Path, log_path: Path,
          rscript: str | None = None, runner: Path | None = None) -> float:
    """Run did_runner.R for the rows of `runs`; returns wall time in seconds. Raises if R fails as a whole
    (a single failed run only writes <run_id>_error.txt, which read_run raises on).

    R reads a script while running it, so the runner is copied into work_dir first: editing utils/r/did_runner.R
    later cannot break a job that is already running.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    runs_path = work_dir / f"runs_{stamp}_{int(time.time() * 1000) % 1000:03d}.csv"
    runs.to_csv(runs_path, index=False)
    script = work_dir / f"did_runner_{runs_path.stem[5:]}.R"
    shutil.copyfile(runner or RUNNER, script)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.run([rscript or find_rscript(), str(script), str(runs_path), str(long_path), str(cells_path), str(out_dir)],
                              stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"R did runner failed (exit {proc.returncode}); see {log_path}")
    return time.time() - t0


def read_run(out_dir: Path, run_id: str) -> dict:
    p = out_dir / run_id
    err = Path(f"{p}_error.txt")
    if err.exists():
        raise RuntimeError(f"R did run {run_id} failed: {err.read_text(encoding='utf-8').strip()}")
    meta = pd.read_csv(f"{p}_meta.csv").iloc[0]
    n, E = int(meta["n"]), int(meta["E"])
    ids = np.fromfile(f"{p}_ids.bin", dtype=np.int32)
    if_simple = np.fromfile(f"{p}_if_simple.bin", dtype=np.float64)
    if_dyn = np.fromfile(f"{p}_if_dynamic.bin", dtype=np.float64).reshape((n, E), order="F")
    agg = pd.read_csv(f"{p}_agg.csv")
    return {
        "gt": pd.read_csv(f"{p}_gt.csv"), "agg": agg, "ids": ids, "if_simple": if_simple, "if_dynamic": if_dyn,
        "event_times": agg.loc[agg["type"] == "dynamic", "e"].astype(int).to_numpy(), "n": n,
        "seconds": int(meta["seconds"]), "warnings": Path(f"{p}_warnings.txt").read_text(encoding="utf-8").splitlines(),
    }


def remove_run_files(out_dir: Path, run_id: str) -> None:
    for suffix in ["_ids.bin", "_if_simple.bin", "_if_dynamic.bin", "_error.txt"]:
        (out_dir / f"{run_id}{suffix}").unlink(missing_ok=True)


def _combine(parts: list[dict], n_cells: int, union: np.ndarray) -> dict:
    """theta = sum_k w_k theta_k with w_k = a_k p_k / sum_h a_h p_h (p_k = cohort share of the union sample, a_k = number of
    (g,t) cells it contributes), and its influence function including the estimated weights, as did::aggte (wif)."""
    N = float(union.sum())
    p = np.array([q["n_treated"] / N for q in parts])
    a = np.array([q["a"] for q in parts], dtype=float)
    th = np.array([q["att"] for q in parts])
    S = float(np.sum(a * p))
    w = a * p / S
    att = float(np.sum(w * th))
    psi = np.zeros(n_cells)
    for k, q in enumerate(parts):
        psi += w[k] * q["psi"]
        # weight term: (1 / (N S)) a_k (theta_k - theta) (D_k - p_k), over the union sample
        dk = -np.full(n_cells, p[k]) * union
        dk[q["treated_pos"]] += 1.0
        psi += a[k] * (th[k] - att) * dk / (N * S)
    return {"att": att, "psi": psi, "weights": w}


def combine_cohort_runs(cohort_runs: list[dict], *, n_cells: int, g_did: np.ndarray) -> dict:
    """Aggregate per-cohort R `did` runs (stacked design: one run per cohort, each with its own control cells) exactly as
    did::aggte aggregates ATT(g,t) from a joint run: `simple` = cohort-size weighted mean of the post-treatment ATT(g,t)
    (t >= g), `dynamic` = cohort-size weighted mean of ATT(g, g+e) by event time e. Influence functions (psi, with
    theta_hat - theta ~ sum_i psi_i) add each cohort's own and the estimation of the cohort weights; a cell used in several
    cohorts' runs (a shared control) gets the sum of its contributions, so any HAC on psi accounts for that overlap.

    cohort_runs: list of {"g": cohort, "pos": positions (0..n_cells-1) of the run's ids in IF row order, "run": read_run(...)}.
    g_did: treatment year by position (> 0 only for the cells treated in some run).
    """
    union = np.zeros(n_cells, dtype=bool)
    for c in cohort_runs:
        union[c["pos"]] = True
    simple_parts, dyn_parts, rows = [], {}, []
    for c in cohort_runs:
        run, pos = c["run"], c["pos"]
        treated_pos = pos[g_did[pos] > 0]
        assert (g_did[treated_pos] == c["g"]).all(), "a run treats cells of another cohort"
        gt = run["gt"]
        n_post = int(((gt["time"] >= gt["group"]) & gt["att"].notna()).sum())
        simple = run["agg"].query("type == 'simple'").iloc[0]
        psi = np.zeros(n_cells)
        psi[pos] = run["if_simple"] / run["n"]
        base = {"g": c["g"], "n_treated": len(treated_pos), "treated_pos": treated_pos}
        if n_post > 0 and np.isfinite(simple["att"]):
            simple_parts.append({**base, "a": n_post, "att": float(simple["att"]), "psi": psi})
        dyn = run["agg"].query("type == 'dynamic'").set_index("e")
        for k, e in enumerate(run["event_times"]):
            att_e = float(dyn.loc[e, "att"])
            if not np.isfinite(att_e):
                continue
            pe = np.zeros(n_cells)
            pe[pos] = run["if_dynamic"][:, k] / run["n"]
            dyn_parts.setdefault(int(e), []).append({**base, "a": 1, "att": att_e, "psi": pe})
        rows.append({"cohort": c["g"], "n_treated": len(treated_pos), "n_units": run["n"], "n_post_cells": n_post,
                     "att": float(simple["att"]), "se_analytic": float(simple["se"])})
    out = {"simple": _combine(simple_parts, n_cells, union), "dynamic": {}, "cohorts": pd.DataFrame(rows), "n_union": int(union.sum())}
    for e in sorted(dyn_parts):
        out["dynamic"][e] = _combine(dyn_parts[e], n_cells, union)
    w = dict(zip([q["g"] for q in simple_parts], out["simple"]["weights"]))
    out["cohorts"]["weight_simple"] = out["cohorts"]["cohort"].map(w).fillna(0.0)
    return out
