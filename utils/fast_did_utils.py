"""Vectorised Callaway-Sant'Anna ATT(g,t) estimation and exact spatial-HAC inference.

Numerically the same estimators as estimation_workflow_utils.did_influence_for_group_time
(no covariates) and dr_did_influence_for_group_time + fit_cohort_propensity_score (doubly
robust plug-in), but organised for speed:

* the outcome is held as one (cells x years) numpy matrix, so every ATT(g,t) of a cohort is
  computed in one vectorised step instead of a pandas pivot per (g,t) cell;
* the DR outcome regression is one multi-output RidgeCV per cohort (alpha chosen per target by
  the same closed-form leave-one-out criterion, so identical to one fit per t), and the
  propensity score is fit once per cohort and control group and cached across anticipation
  settings (covariates are time-invariant cell attributes);
* influence functions are aggregated on the fly into the overall ATT and the event-time ATTs,
  with the same cohort-size weights as the point estimates, so the reported SEs include the
  covariance across (g,t) cells (they share controls and treated cells);
* the Conley spatial-HAC variance psi' K psi is computed exactly with an FFT convolution on the
  1 km lattice, without any pair cap (the pair-loop implementation truncated at max_pairs).

Influence-function scaling follows the existing code: Var(estimate) ~ sum_i psi_i^2 under
independence, so psi of an aggregate is the weighted sum of the (g,t) psi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


# --------------------------------------------------------------------------------------------
# Data layout
# --------------------------------------------------------------------------------------------
def outcome_matrix(panel: pd.DataFrame, cell_col: str, year_col: str, outcome_col: str):
    """(cell ids, years, Y) with Y[i, j] = outcome of cell i in years[j]."""
    years = np.sort(panel[year_col].unique().astype(int))
    cells, cell_idx = np.unique(panel[cell_col].to_numpy(), return_inverse=True)
    year_idx = np.searchsorted(years, panel[year_col].to_numpy().astype(int))
    Y = np.full((len(cells), len(years)), np.nan)
    Y[cell_idx, year_idx] = panel[outcome_col].to_numpy(dtype=float)
    return cells, years, Y


# --------------------------------------------------------------------------------------------
# Propensity score (same specification as fit_cohort_propensity_score)
# --------------------------------------------------------------------------------------------
def fit_propensity(x: np.ndarray, d: np.ndarray, cv: int = 5, trim: float = 0.02) -> dict:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    n_treated, n_control = int(d.sum()), int((1 - d).sum())
    if n_treated == 0 or n_control < x.shape[1] + 2 or np.unique(d).size < 2:
        return {"scaler": None, "model": None, "n_treated": n_treated, "n_control": n_control, "share_trimmed": np.nan}
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    n_splits = min(int(cv), int(np.bincount(d.astype(int)).min()))
    if n_splits < 2:
        model = LogisticRegression(C=1.0, max_iter=2000).fit(xs, d)
    else:
        model = LogisticRegressionCV(Cs=10, cv=n_splits, scoring="neg_log_loss", max_iter=2000, n_jobs=-1).fit(xs, d)
    p = model.predict_proba(xs)[:, 1]
    return {"scaler": scaler, "model": model, "n_treated": n_treated, "n_control": n_control,
            "share_trimmed": float(np.mean((p < trim) | (p > 1 - trim)))}


def predict_propensity(ps: dict, x: np.ndarray) -> np.ndarray | None:
    if ps is None or ps.get("model") is None:
        return None
    return ps["model"].predict_proba(ps["scaler"].transform(x))[:, 1]


# --------------------------------------------------------------------------------------------
# ATT(g,t) for one cohort, all t at once
# --------------------------------------------------------------------------------------------
def _nocov_block(dY: np.ndarray, d: np.ndarray):
    """dY: (n x T) long differences for treated (d=1) and control (d=0) rows."""
    tr, co = d == 1, d == 0
    nT, nC = int(tr.sum()), int(co.sum())
    mT, mC = dY[tr].mean(axis=0), dY[co].mean(axis=0)
    att = mT - mC
    psi = np.empty_like(dY)
    psi[tr] = (dY[tr] - mT) / nT
    psi[co] = -(dY[co] - mC) / nC
    return att, psi


def _dr_block(dY: np.ndarray, d: np.ndarray, x: np.ndarray, p_hat: np.ndarray, trim: float, max_weight: float):
    """Sant'Anna-Zhao traditional DR-DID plug-in, vectorised over the columns of dY."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    co = d == 0
    or_scaler = StandardScaler().fit(x[co])
    xs = or_scaler.transform(x)
    alpha_per_target = dY.shape[1] > 1
    or_model = RidgeCV(alphas=np.logspace(-3, 4, 30), alpha_per_target=alpha_per_target).fit(
        xs[co], dY[co] if alpha_per_target else dY[co, 0])
    m0 = or_model.predict(xs)
    if m0.ndim == 1:
        m0 = m0[:, None]
    p = np.clip(p_hat, trim, 1 - trim)
    w1 = d.astype(float)
    w0 = np.minimum(p * (1 - d) / (1 - p), max_weight)
    n = len(d)
    e1, e0 = w1.mean(), w0.mean()
    resid = dY - m0
    tau1 = (w1[:, None] * resid).mean(axis=0) / e1
    tau0 = (w0[:, None] * resid).mean(axis=0) / e0
    psi = (w1[:, None] * (resid - tau1)) / (e1 * n) - (w0[:, None] * (resid - tau0)) / (e0 * n)
    return tau1 - tau0, psi


def cohort_att_gt(
    Y: np.ndarray,
    years: np.ndarray,
    g_cell: np.ndarray,
    control_mask: np.ndarray,
    cohort: int,
    *,
    anticipation: int = 0,
    control_type: str = "never",
    x: np.ndarray | None = None,
    x_complete: np.ndarray | None = None,
    ps_cache: dict | None = None,
    ps_key=None,
    ps_cv: int = 5,
    ps_trim: float = 0.02,
    ps_max_weight: float = 100.0,
    notyet_ok: np.ndarray | None = None,
):
    """All ATT(g,t) of one cohort (t != base year) with their influence functions.

    g_cell: treatment year per cell (0 = never treated). control_mask: never-treated cells in the
    chosen control group (control_type='never'), or ignored for control_type='notyet' where the
    controls at (g,t) are the eventually-treated cells with g' > max(t, base) + anticipation.
    notyet_ok: optional (cells x years) boolean eligibility for not-yet-treated controls, evaluated
    at the year max(t, base); e.g. "more than 35 km from every cell protected by that year".
    x: (cells x k) covariates for the DR estimator (None = no covariates). Returns a list of
    dicts {time, att, rows (cell indices), psi (values)} and the propensity diagnostics.
    """
    base = cohort - 1 - int(anticipation)
    if base not in years:
        return [], None
    jb = int(np.searchsorted(years, base))
    t_cols = [j for j in range(len(years)) if j != jb]
    treated = g_cell == cohort
    if x is not None:
        treated = treated & x_complete
    out = []

    def finish(rows, d, cols, att, psi):
        for k, j in enumerate(cols):
            out.append({"group": int(cohort), "time": int(years[j]), "att": float(att[k]), "rows": rows, "psi": psi[:, k], "n_treated": int(d.sum()), "n_control": int((1 - d).sum())})

    if control_type == "never":
        ctrl = control_mask.copy()
        if x is not None:
            ctrl &= x_complete
        rows = np.flatnonzero(treated | ctrl)
        d = treated[rows].astype(float)
        if d.sum() == 0 or (1 - d).sum() == 0:
            return [], None
        dY = Y[rows][:, t_cols] - Y[rows, jb][:, None]
        if x is None:
            att, psi = _nocov_block(dY, d)
            finish(rows, d, t_cols, att, psi)
            return out, None
        key = (ps_key, cohort)
        if ps_cache is not None and key in ps_cache:
            ps = ps_cache[key]
        else:
            ps = fit_propensity(x[rows], d, cv=ps_cv, trim=ps_trim)
            if ps_cache is not None:
                ps_cache[key] = ps
        p_hat = predict_propensity(ps, x[rows])
        if p_hat is None:
            return [], ps
        att, psi = _dr_block(dY, d, x[rows], p_hat, ps_trim, ps_max_weight)
        finish(rows, d, t_cols, att, psi)
        return out, ps

    if control_type == "notyet":
        ps = None
        if x is not None:
            # propensity fit once per cohort on the controls available at the base year
            ctrl0 = (g_cell > base + int(anticipation)) & (g_cell != cohort) & (g_cell > 0) & x_complete
            if notyet_ok is not None:
                ctrl0 &= notyet_ok[:, jb]
            rows0 = np.flatnonzero(treated | ctrl0)
            key = (ps_key, cohort)
            if ps_cache is not None and key in ps_cache:
                ps = ps_cache[key]
            else:
                ps = fit_propensity(x[rows0], treated[rows0].astype(float), cv=ps_cv, trim=ps_trim)
                if ps_cache is not None:
                    ps_cache[key] = ps
        for j in t_cols:
            limit = max(int(years[j]), base) + int(anticipation)
            ctrl = (g_cell > limit) & (g_cell != cohort) & (g_cell > 0)
            if notyet_ok is not None:
                ctrl &= notyet_ok[:, int(np.searchsorted(years, max(int(years[j]), base)))]
            if x is not None:
                ctrl &= x_complete
            rows = np.flatnonzero(treated | ctrl)
            d = treated[rows].astype(float)
            if d.sum() == 0 or (1 - d).sum() < 2:
                continue
            dY = (Y[rows, j] - Y[rows, jb])[:, None]
            if x is None:
                att, psi = _nocov_block(dY, d)
            else:
                p_hat = predict_propensity(ps, x[rows])
                if p_hat is None:
                    continue
                att, psi = _dr_block(dY, d, x[rows], p_hat, ps_trim, ps_max_weight)
            finish(rows, d, [j], att, psi)
        return out, ps
    raise ValueError("control_type must be 'never' or 'notyet'")


# --------------------------------------------------------------------------------------------
# Aggregation with influence functions
# --------------------------------------------------------------------------------------------
class GTAccumulator:
    """Folds each cohort's ATT(g,t) and influence functions into the overall ATT and the
    event-time ATTs as soon as they are computed, so only one cohort's influence matrix is in
    memory at a time.

    Weights follow aggregate_group_time_atts: overall = cohort-size-weighted mean of the
    post-treatment ATT(g,t) (t >= g - anticipation); event time e = cohort-size-weighted mean of
    ATT(g, g+e). Weights are normalised over the (g,t) cells that were actually estimated, and
    the aggregated influence vector is the same weighted sum of the (g,t) influence vectors, so
    its variance includes the covariance across (g,t) cells.
    """

    def __init__(self, cohort_sizes: dict, n_cells: int, *, anticipation: int = 0, event_window=(-8, 5)):
        self.sizes = {int(k): float(v) for k, v in cohort_sizes.items()}
        self.n = int(n_cells)
        self.antic = int(anticipation)
        self.ew = tuple(event_window)
        self.rows = []
        self.ov_num, self.ov_w = 0.0, 0.0
        self.ov_psi = np.zeros(self.n)
        self.ev = {}

    def add(self, results: list[dict]):
        for r in results:
            g, t = r["group"], r["time"]
            w = self.sizes.get(g, 0.0)
            e = t - g
            self.rows.append({k: r[k] for k in ("group", "time", "att", "n_treated", "n_control")} | {"event_time": e, "w": w})
            if w <= 0:
                continue
            if t >= g - self.antic:
                self.ov_num += w * r["att"]
                self.ov_w += w
                np.add.at(self.ov_psi, r["rows"], w * r["psi"])
            if self.ew[0] <= e <= self.ew[1]:
                ent = self.ev.setdefault(e, {"num": 0.0, "w": 0.0, "psi": np.zeros(self.n), "cohorts": 0})
                ent["num"] += w * r["att"]
                ent["w"] += w
                ent["cohorts"] += 1
                np.add.at(ent["psi"], r["rows"], w * r["psi"])

    def result(self):
        gt = pd.DataFrame(self.rows)
        overall = None
        if self.ov_w > 0:
            overall = {"att": self.ov_num / self.ov_w, "psi": self.ov_psi / self.ov_w,
                       "n_gt": int(((gt["time"] >= gt["group"] - self.antic) & (gt["w"] > 0)).sum())}
        events = {e: {"att": v["num"] / v["w"], "psi": v["psi"] / v["w"], "n_cohorts": v["cohorts"]}
                  for e, v in sorted(self.ev.items()) if v["w"] > 0}
        return gt, overall, events


# --------------------------------------------------------------------------------------------
# Exact Conley spatial-HAC variance via FFT
# --------------------------------------------------------------------------------------------
class ConleyLattice:
    """Maps cells to a regular lattice once; variance(psi) = psi' K psi with K the Bartlett
    (or uniform) kernel of distance, cutoff `cutoff_km`. Cells sharing a lattice node are
    treated as distance 0. The default 500 m lattice gives one node per 1 km cell (no collisions) and matches
    the exact pair sum to within 0.01% (checked against spatial_hac_variance without a pair cap)."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, *, cutoff_km: float = 35.0, res_m: float = 500.0, kernel: str = "bartlett"):
        self.ix = np.rint((x_m - np.nanmin(x_m)) / res_m).astype(int)
        self.iy = np.rint((y_m - np.nanmin(y_m)) / res_m).astype(int)
        self.shape = (int(self.iy.max()) + 1, int(self.ix.max()) + 1)
        R = int(np.ceil(cutoff_km * 1000.0 / res_m))
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        dist_km = np.hypot(ox, oy) * res_m / 1000.0
        if kernel == "bartlett":
            K = np.clip(1.0 - dist_km / cutoff_km, 0.0, None)
        elif kernel == "uniform":
            K = (dist_km <= cutoff_km).astype(float)
        else:
            raise ValueError("kernel must be 'bartlett' or 'uniform'")
        self.K = K
        flat = self.iy * self.shape[1] + self.ix
        self.n_collisions = int(len(flat) - len(np.unique(flat)))

    def variance(self, psi: np.ndarray) -> float:
        G = np.zeros(self.shape)
        np.add.at(G, (self.iy, self.ix), psi)
        return float(max(np.sum(G * fftconvolve(G, self.K, mode="same")), 0.0))

    def se(self, psi: np.ndarray) -> float:
        return float(np.sqrt(self.variance(psi)))
