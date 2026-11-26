"""Out-of-sample checks of the stage-1 trend model.

For each cutoff date we refit using only data released by then and score the model's predictive
distribution for every later METR frontier measurement:
  - log predictive density (higher is better), averaged over held-out points
  - coverage of the central 50% and 80% predictive intervals (well calibrated ~ 0.5 and 0.8)
Variants compare the full regime mixture with exponential-only and superexponential-only priors.

Caveat: METR re-measured older models when it moved to v1.1, so "data available at the cutoff"
uses today's measurements of old models, not the numbers published then.
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from .data import frontier_observations, metr_points
from .model import Priors, fit, log2h_at, student_t_logpdf

VARIANTS = {
    "mixture (used)": {},
    "exponential only": {"regime_prior": {"super": 1e-9, "exp": 1.0, "sub": 1e-9}},
    "superexponential only": {"regime_prior": {"super": 1.0, "exp": 1e-9, "sub": 1e-9}},
}


def heldout_points(cutoff: float, start: float = 2023.0):
    m = metr_points().sort_values("t")
    m = m[m["t"] >= start]
    front = m[m["y"] >= m["y"].cummax() - 1e-12]
    return front[front["t"] > cutoff]


def score(cutoff: float, variant: dict, n: int = 60_000, draws: int = 4000, seed: int = 0) -> dict:
    pr = replace(Priors(), **variant)
    obs, _ = frontier_observations(cutoff=cutoff, use_nowcast=False)
    post = fit(obs, pr, n=n, seed=seed, pmc_rounds=1)
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(len(post.w), size=draws, p=post.w)
    held = heldout_points(cutoff)
    lpd, in50, in80, rows = [], [], [], []
    for _, r in held.iterrows():
        mu = log2h_at(r["t"], post.y_a[idx], post.D_a[idx], post.g[idx])
        scale = np.sqrt(r["se"] ** 2 + post.sigma_x[idx] ** 2)
        ok = np.isfinite(mu)
        dens = np.zeros(draws)
        dens[ok] = np.exp(student_t_logpdf(r["y"] - mu[ok], scale[ok], pr.t_df))
        lpd.append(float(np.log(dens.mean() + 1e-300)))
        # predictive draws for coverage
        pred = mu + scale * rng.standard_t(pr.t_df, size=draws)
        pred = pred[np.isfinite(pred)]
        q = np.quantile(pred, [0.1, 0.25, 0.75, 0.9]) if pred.size else [np.nan] * 4
        in50.append(q[1] <= r["y"] <= q[2]); in80.append(q[0] <= r["y"] <= q[3])
        rows.append({"model": r["name"], "t": round(float(r["t"]), 3), "y": round(float(r["y"]), 3),
                     "pred_median": round(float(np.median(pred)), 3) if pred.size else None,
                     "pred_lo80": round(float(q[0]), 3), "pred_hi80": round(float(q[3]), 3)})
    return {"cutoff": cutoff, "n_train": len(obs.t), "n_test": len(held),
            "mean_lpd": float(np.mean(lpd)) if lpd else None,
            "cover50": float(np.mean(in50)) if in50 else None, "cover80": float(np.mean(in80)) if in80 else None,
            "regime_post": post.regime_post, "points": rows}


def run(cutoffs=(2024.75, 2025.0, 2025.25, 2025.5, 2025.75)) -> dict:
    out = {}
    for name, v in VARIANTS.items():
        res = [score(c, v) for c in cutoffs]
        res = [r for r in res if r["n_test"]]
        tot = sum(r["n_test"] for r in res)
        out[name] = {
            "by_cutoff": res,
            "mean_lpd": sum(r["mean_lpd"] * r["n_test"] for r in res) / tot,
            "cover50": sum(r["cover50"] * r["n_test"] for r in res) / tot,
            "cover80": sum(r["cover80"] * r["n_test"] for r in res) / tot,
            "n_scored": tot,
        }
    return out


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_cutoff"} for k, v in r.items()}, indent=1))
    for c in r["mixture (used)"]["by_cutoff"]:
        print(c["cutoff"], c["n_train"], c["n_test"], round(c["mean_lpd"], 3), c["regime_post"])
