"""Turn the raw downloads into the observations the model fits.

Two observation streams feed the frontier time-horizon trend:

1. METR-measured 50% time horizons for frontier models (direct measurements, with CIs).
2. "Nowcast" horizons for frontier models METR hasn't measured yet. METR's public data lags
   by months, so we calibrate a linear map log2(horizon) = a + b * ECI on models that have both
   a METR horizon and an Epoch Capabilities Index (ECI) score, then translate the newer ECI
   frontier into horizons. Each nowcast carries the full predictive uncertainty of that map.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
Z95 = 1.959964


def year_frac(d) -> float:
    d = pd.Timestamp(d)
    start = pd.Timestamp(year=d.year, month=1, day=1)
    end = pd.Timestamp(year=d.year + 1, month=1, day=1)
    return d.year + (d - start) / (end - start)


def from_year_frac(y: float) -> pd.Timestamp:
    year = int(math.floor(y))
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return start + (end - start) * (y - year)


@dataclass
class Obs:
    """Frontier observations of log2(50% horizon in minutes) with standard errors."""
    t: np.ndarray          # decimal year
    y: np.ndarray          # log2 horizon (minutes)
    se: np.ndarray         # standard error of y
    kind: list[str]        # "metr" or "nowcast"
    name: list[str]


def metr_points(raw: Path = RAW) -> pd.DataFrame:
    doc = yaml.safe_load((raw / "metr_horizons.yaml").read_text())
    rows = []
    for name, v in doc["results"].items():
        m = v.get("metrics", v)
        p50 = m.get("p50_horizon_length") or {}
        if not p50.get("estimate") or not v.get("release_date"):
            continue
        lo, hi = p50.get("ci_low"), p50.get("ci_high")
        se = (math.log2(hi) - math.log2(lo)) / (2 * Z95) if lo and hi and lo > 0 else 1.0
        rows.append({
            "name": name, "date": pd.Timestamp(v["release_date"]), "minutes": p50["estimate"],
            "y": math.log2(p50["estimate"]), "se": se, "is_sota": bool(m.get("is_sota", False)),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["t"] = df["date"].map(year_frac)
    return df


def eci_points(raw: Path = RAW) -> pd.DataFrame:
    e = pd.read_csv(raw / "eci_scores.csv")
    e["date"] = pd.to_datetime(e["date"])
    e = e.dropna(subset=["eci", "date"]).sort_values("date").reset_index(drop=True)
    e["se"] = (e["eci_ci_high"] - e["eci_ci_low"]) / (2 * Z95)
    e["t"] = e["date"].map(year_frac)
    return e


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"_?(inspect|unknown|16k|high|medium|xhigh|low)$", "", s)
    s = re.sub(r"\(.*?\)", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def matched_metr_eci(raw: Path = RAW) -> pd.DataFrame:
    """Models with both a METR horizon and an ECI score, joined on organization + release date,
    breaking ties by name similarity. Epoch's copy of the METR table carries the organization."""
    t = pd.read_csv(raw / "epoch_benchmarks" / "metr_time_horizons_external.csv")
    t = t.dropna(subset=["Time horizon", "Release date"])
    t["date"] = pd.to_datetime(t["Release date"])
    t["y"] = np.log2(t["Time horizon"])
    t["se"] = (np.log2(t["CI_high"]) - np.log2(t["CI_low"])) / (2 * Z95)
    # several reasoning-effort settings per model: keep the best-performing one
    t["base"] = t["Model version"].map(_norm)
    t = t.sort_values("y", ascending=False).drop_duplicates(["Organization", "date"])
    e = eci_points(raw)
    rows = []
    for _, r in t.iterrows():
        cand = e[(e["Organization"] == r["Organization"]) & ((e["date"] - r["date"]).abs() <= pd.Timedelta(days=3))]
        if cand.empty:
            continue
        score = cand["Model"].map(lambda m: difflib.SequenceMatcher(None, _norm(m), r["base"]).ratio())
        best = cand.loc[score.idxmax()]
        if score.max() < 0.45:
            continue
        rows.append({"metr_model": r["Model version"], "eci_model": best["Model"], "date": r["date"],
                     "y": r["y"], "y_se": r["se"], "eci": best["eci"], "eci_se": best["se"]})
    return pd.DataFrame(rows)


@dataclass
class EciMap:
    a: float
    b: float
    cov: np.ndarray   # 2x2 covariance of (a, b)
    sigma: float      # residual scatter (model-to-model) in log2 units
    n: int

    def predict(self, eci: np.ndarray, eci_se: np.ndarray):
        X = np.stack([np.ones_like(eci), eci], axis=1)
        mean = self.a + self.b * eci
        var = np.einsum("ij,jk,ik->i", X, self.cov, X) + self.sigma ** 2 + (self.b * eci_se) ** 2
        return mean, np.sqrt(var)


def fit_eci_map(m: pd.DataFrame) -> EciMap:
    """Weighted least squares of log2 horizon on ECI. The residual scatter sigma is estimated
    by the method of moments so the weights include it (iterated twice)."""
    x, y = m["eci"].to_numpy(), m["y"].to_numpy()
    se2 = m["y_se"].to_numpy() ** 2
    sigma2 = 0.25
    for _ in range(20):
        w = 1.0 / (se2 + sigma2)
        X = np.stack([np.ones_like(x), x], axis=1)
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        beta = cov @ (X.T @ (w * y))
        resid = y - X @ beta
        # DerSimonian-Laird style update of the extra variance
        new = max(0.0, (np.sum(resid ** 2) - np.sum(se2)) / max(len(x) - 2, 1))
        if abs(new - sigma2) < 1e-6:
            break
        sigma2 = new
    # METR's CIs are wide enough that the moment estimate of model-to-model scatter often hits 0,
    # which would make nowcasts overconfident. Floor it at 0.25 doublings (a judgment call: the
    # ECI-horizon relation is not exact across labs and scaffolds).
    return EciMap(a=float(beta[0]), b=float(beta[1]), cov=cov, sigma=max(math.sqrt(sigma2), SIGMA_FLOOR), n=len(x))


SIGMA_FLOOR = 0.25


def frontier_observations(cutoff: float | None = None, start: float = 2023.0, raw: Path = RAW,
                          use_nowcast: bool = True) -> tuple[Obs, dict]:
    """Frontier horizon observations up to `cutoff` (decimal year; None = all data).

    For backtests, `cutoff` hides everything released after it, including the ECI map's
    calibration points, so the model only sees what was knowable then (up to later revisions
    of the source datasets, which we can't undo)."""
    metr = metr_points(raw)
    if cutoff is not None:
        metr = metr[metr["t"] <= cutoff]
    # running frontier: a model counts if it beats every earlier METR model
    metr = metr.sort_values("t")
    metr = metr[metr["y"] >= metr["y"].cummax() - 1e-12]
    metr = metr[metr["t"] >= start]
    t, y, se = list(metr["t"]), list(metr["y"]), list(metr["se"])
    kind, name = ["metr"] * len(t), list(metr["name"])
    info = {"n_metr": len(t), "metr_last": float(metr["t"].max()) if len(metr) else None}

    if use_nowcast:
        matched = matched_metr_eci(raw)
        if cutoff is not None:
            matched = matched[matched["date"].map(year_frac) <= cutoff]
        if len(matched) >= 6:
            emap = fit_eci_map(matched)
            eci = eci_points(raw)
            if cutoff is not None:
                eci = eci[eci["t"] <= cutoff]
            last_metr = info["metr_last"] or start
            eci = eci.sort_values("t")
            frontier = eci[eci["eci"] >= eci["eci"].cummax() - 1e-12]
            new = frontier[frontier["t"] > last_metr + 1 / 52]
            if len(new):
                mu, sd = emap.predict(new["eci"].to_numpy(), new["se"].to_numpy())
                t += list(new["t"]); y += list(mu); se += list(sd)
                kind += ["nowcast"] * len(new); name += list(new["Model"])
            info.update({"eci_map": {"a": emap.a, "b": emap.b, "sigma": emap.sigma, "n": emap.n},
                         "n_nowcast": int(len(new))})
    order = np.argsort(t)
    obs = Obs(t=np.array(t)[order], y=np.array(y)[order], se=np.array(se)[order],
              kind=[kind[i] for i in order], name=[name[i] for i in order])
    return obs, info
