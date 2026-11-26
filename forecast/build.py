"""Run the whole pipeline and write the JSON the website reads.

    python -m forecast.build            # fit, simulate, write site/data/forecast.json, append history
    python -m forecast.build --history  # also rebuild the back-filled history (slow)
    python -m forecast.build --backtest # also rerun the out-of-sample backtest (slow)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RAW, frontier_observations, year_frac, from_year_frac
from .model import Priors, REGIMES, T_ANCHOR, event_log_lr, fit, log2h_at, simulate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data"
PROCESSED = ROOT / "data" / "processed"
MILESTONES = {"ac": "Automated coder", "sar": "Superhuman AI researcher",
              "siar": "Superintelligent AI researcher", "asi": "Artificial superintelligence"}


def iso(y: float) -> str | None:
    if not np.isfinite(y) or y > 2200:
        return None
    return from_year_frac(float(y)).strftime("%Y-%m-%d")


def summarize(a: np.ndarray, now: float) -> dict:
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    vals = np.quantile(a, qs)
    years = list(range(int(np.floor(now)), 2061))
    return {
        "quantiles": {str(q): float(v) if np.isfinite(v) else None for q, v in zip(qs, vals)},
        "quantile_dates": {str(q): iso(v) for q, v in zip(qs, vals)},
        "p_by_year_end": {str(y): float(np.mean(a < y + 1)) for y in years},
        "p_after_2060": float(np.mean(a >= 2061)),
    }


def density(a: np.ndarray, lo: float, hi: float, step: float = 0.25) -> dict:
    edges = np.arange(lo, hi + 1e-9, step)
    h, _ = np.histogram(a, bins=edges)
    return {"start": lo, "step": step, "prob": (h / len(a)).round(6).tolist(),
            "before": float(np.mean(a < lo)), "after": float(np.mean(a >= hi))}


def horizon_fan(post, obs, t0=2023.0, t1=2032.0, draws=6000, seed=3) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post.w), size=draws, p=post.w)
    ts = np.round(np.arange(t0, t1 + 1e-9, 1 / 12), 4)
    bands = {k: [] for k in ("p10", "p25", "p50", "p75", "p90")}
    for t in ts:
        y = log2h_at(t, post.y_a[idx], post.D_a[idx], post.g[idx])
        y = np.where(np.isfinite(y), y, 60.0)  # past a blow-up: off the chart
        for k, q in zip(bands, (0.1, 0.25, 0.5, 0.75, 0.9)):
            bands[k].append(round(float(np.quantile(y, q)), 3))
    return {"t": ts.tolist(), **bands,
            "obs": [{"t": round(float(t), 4), "y": round(float(y), 3), "se": round(float(s), 3), "kind": k, "name": n}
                    for t, y, s, k, n in zip(obs.t, obs.y, obs.se, obs.kind, obs.name)]}


def run_once(now: float, pr: Priors = Priors(), cutoff: float | None = None, use_events=True,
             use_nowcast=True, n=200_000, m=40_000, seed=0):
    obs, info = frontier_observations(cutoff=cutoff, use_nowcast=use_nowcast)
    lr, used = event_log_lr(cutoff) if use_events else (np.zeros(3), [])
    post = fit(obs, pr, n=n, seed=seed, log_lr_by_regime=lr)
    sim = simulate(post, now, pr, m=m, seed=seed + 1)
    return obs, info, post, sim, used


def weighted_q(x, w, qs=(0.1, 0.5, 0.9)):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return [float(x[o][min(np.searchsorted(c, q), len(x) - 1)]) for q in qs]


def build_history(now: float, start="2024-07-01") -> list:
    """What this model would have said on the first of each month, using only data released by
    then (and only events dated by then). A pseudo-history: today's dataset versions are used."""
    out = []
    for d in pd.date_range(start, pd.Timestamp(from_year_frac(now)), freq="MS"):
        c = year_frac(d)
        try:
            _, info, post, sim, used = run_once(c, cutoff=c, n=60_000, m=15_000)
        except Exception as e:  # too little data at early cutoffs
            print("history skip", d.date(), e)
            continue
        out.append(history_row(d.strftime("%Y-%m-%d"), sim, post, info, len(used), kind="backfill"))
        print("history", d.date(), round(float(np.median(sim.asi)), 2))
    return out


def history_row(date: str, sim, post, info, n_events, kind="live") -> dict:
    row = {"date": date, "kind": kind, "n_events": n_events, "n_obs": info.get("n_metr", 0) + info.get("n_nowcast", 0),
           "regime_post": {k: round(v, 4) for k, v in post.regime_post.items()}}
    for k in MILESTONES:
        a = getattr(sim, k)
        row[k] = {q: round(float(np.quantile(a, float(q))), 3) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")}
    return row


def sensitivity(now: float) -> list:
    base = Priors()
    cases = [
        ("Main model", {}, {}),
        ("Without capability events", {}, {"use_events": False}),
        ("Without ECI nowcasts (METR data only)", {}, {"use_nowcast": False}),
        ("Exponential trend only", {"regime_prior": {"super": 1e-9, "exp": 1.0, "sub": 1e-9}}, {}),
        ("Superexponential trend only", {"regime_prior": {"super": 1.0, "exp": 1e-9, "sub": 1e-9}}, {}),
        ("Harder AC threshold (10x longer horizon)", {"thr_minutes_ci80": (100_000.0, 48_000_000.0)}, {}),
        ("Easier AC threshold (10x shorter horizon)", {"thr_minutes_ci80": (1_000.0, 480_000.0)}, {}),
        ("Slower takeoff (AI R&D multipliers / 3)", {"m_sc_ci90": (1.5, 4.0), "m_sar_ci90": (3.0, 27.0)}, {}),
        ("AI-2027's original SIAR->ASI tail (to 1,000,000 years)", {"w_siar_asi_lo80": 95.0 ** 2 / 1_000_000.0 ** 1}, {}),
    ]
    rows = []
    for name, pchange, kw in cases:
        pr = replace(base, **pchange)
        if "SIAR->ASI tail" in name:
            # reproduce AI-2027's asymmetric interval with a lognormal through (2.4, 1e6): median sqrt
            pr = replace(base, w_siar_asi_median=float(np.sqrt(2.4 * 1_000_000)), w_siar_asi_lo80=2.4)
        _, _, post, sim, _ = run_once(now, pr, **kw)  # same sample sizes and seeds as the main run
        rows.append({"case": name,
                     "ac": [round(float(np.quantile(sim.ac, q)), 2) for q in (0.1, 0.5, 0.9)],
                     "asi": [round(float(np.quantile(sim.asi, q)), 2) for q in (0.1, 0.5, 0.9)],
                     "p_asi_2030": round(float(np.mean(sim.asi < 2031)), 3),
                     "p_asi_2035": round(float(np.mean(sim.asi < 2036)), 3)})
        print("sensitivity", name, rows[-1]["asi"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--now", default=None, help="YYYY-MM-DD (default: today, UTC)")
    a = ap.parse_args()
    today = pd.Timestamp(a.now) if a.now else pd.Timestamp(datetime.now(timezone.utc).date())
    now = year_frac(today)
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    obs, info, post, sim, used = run_once(now)
    g10, g50, g90 = weighted_q(post.g, post.w)
    D_now = np.quantile(sim.D_now, [0.1, 0.5, 0.9]) * 365.25
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": today.strftime("%Y-%m-%d"),
        "now": now,
        "milestones": {k: {"label": v, **summarize(getattr(sim, k), now)} for k, v in MILESTONES.items()},
        "density": {k: density(getattr(sim, k), np.floor(now), 2061.0) for k in MILESTONES},
        "horizon": horizon_fan(post, obs),
        "trend": {
            "regime_post": post.regime_post,
            "g": [g10, g50, g90],
            "doubling_days_now": D_now.round(1).tolist(),
            "horizon_hours_now": (2 ** np.quantile(sim.y_now, [0.1, 0.5, 0.9]) / 60).round(1).tolist(),
            "ess": post.ess, "log_evidence": post.log_evidence,
        },
        "data": {**info, "sources": json.loads((RAW / "fetch_log.json").read_text())},
        "events": used,
        "priors": asdict(Priors()),
    }
    if a.sensitivity or not (OUT / "sensitivity.json").exists():
        (OUT / "sensitivity.json").write_text(json.dumps({"as_of": result["as_of"], "rows": sensitivity(now)}, indent=1))
    if a.backtest or not (OUT / "backtest.json").exists():
        from .backtest import run as run_bt
        (OUT / "backtest.json").write_text(json.dumps(run_bt(), indent=1))

    hist_path = OUT / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if a.history or not hist:
        hist = [h for h in hist if h["kind"] == "live"] + build_history(now)
    hist = [h for h in hist if not (h["kind"] == "live" and h["date"] == result["as_of"])]
    hist.append(history_row(result["as_of"], sim, post, info, len(used)))
    hist.sort(key=lambda h: (h["date"], h["kind"] != "backfill"))
    hist_path.write_text(json.dumps(hist, indent=1))
    (OUT / "forecast.json").write_text(json.dumps(result, indent=1, default=float))
    m = result["milestones"]
    print("AC median", m["ac"]["quantile_dates"]["0.5"], "| ASI median", m["asi"]["quantile_dates"]["0.5"],
          "| P(ASI by end 2030)", round(m["asi"]["p_by_year_end"].get("2030", float("nan")), 3),
          "| regimes", {k: round(v, 3) for k, v in post.regime_post.items()}, "| ESS", round(post.ess))


if __name__ == "__main__":
    main()
