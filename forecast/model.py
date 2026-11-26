"""Monte Carlo forecast of the recursive-self-improvement inflection and ASI.

Stage 1 (timelines). The frontier 50% task time horizon h(t) (METR) grows with a doubling time
that may itself shrink (superexponential), stay fixed (exponential) or grow (subexponential).
With D_a the doubling time at the anchor date t_a and g the factor by which each successive
doubling time is multiplied, gaining n doublings after t_a takes

    tau(n) = D_a * (1 - g**n) / (1 - g)        (g != 1),     tau(n) = D_a * n   (g == 1),

so log2 h(t) = y_a + n(t) with n(t) = ln(1 - (1 - g) (t - t_a) / D_a) / ln g. For g < 1 the horizon
diverges in finite time, at t_a + D_a / (1 - g). We put a prior on (y_a, D_a, g, sigma_x), with g
drawn from a three-way mixture (super / exponential / sub), and update it on the frontier
observations by importance sampling with a Student-t likelihood (robust to outliers).

The automated-coder (AC, AI-2027's "superhuman coder") date is when log2 h crosses an uncertain
threshold, plus a gap for cost, speed and deployment. We condition on AC not having happened yet.

Stage 2 (takeoff). From AC to superhuman AI researcher (SAR), superintelligent AI researcher
(SIAR) and ASI we follow AI-2027's takeoff forecast: each gap is an amount of human-only research
time W, crossed with an AI R&D progress multiplier that rises from m_A to m_B. With the multiplier
interpolated exponentially in (m - 1) over the fraction f of the gap crossed,
m(f) = 1 + (m_A - 1) r**f with r = (m_B - 1)/(m_A - 1), the calendar time is exactly

    T = W * (1 - ln(m_B / m_A) / ln r)          (T = W / m_A when r == 1).

Unlike AI-2027, the multipliers themselves are uncertain here (lognormal), not point values.

Capability events (e.g. a formally verified proof of a famous open problem) enter as likelihood
ratios over the three growth regimes, documented with sources in events/events.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .data import Obs

Z80 = 1.281552  # z for a two-sided 80% interval
Z90 = 1.644854
REGIMES = ("super", "exp", "sub")
T_ANCHOR = 2026.0


def ln_from_ci(lo: float, hi: float, z: float = Z80) -> tuple[float, float]:
    """(mu, sigma) of a lognormal whose central interval at +-z is [lo, hi]."""
    return (math.log(lo) + math.log(hi)) / 2, (math.log(hi) - math.log(lo)) / (2 * z)


@dataclass
class Priors:
    # stage 1: trend
    y_a_mean: float = 8.5          # log2 minutes at T_ANCHOR (~6 h); weakly informative
    y_a_sd: float = 2.0
    D_a_ci80: tuple = (0.20, 1.00)  # doubling time at anchor, years (METR from-2023: 0.35 y); widened after backtests
    regime_prior: dict = field(default_factory=lambda: {"super": 0.40, "exp": 0.45, "sub": 0.15})
    g_super_ci80: tuple = (0.70, 0.97)   # each doubling 3-30% faster than the last; widened after backtests
    g_sub_ci80: tuple = (1.03, 1.20)
    sigma_x_scale: float = 0.5      # half-normal scale for extra frontier scatter (log2 units); widened after backtests
    t_df: float = 5.0               # Student-t degrees of freedom
    # AC threshold on the 50% horizon (minutes). ~2000 work-hours per work-year.
    # median 2 work-years; 80% CI 1 work-month .. 40 work-years. AI-2027 put the 80%-reliability
    # horizon at 1.5 months (Nikola) to 10 years (Eli); 50% horizons run ~4-6x longer than 80%.
    thr_minutes_ci80: tuple = (10_000.0, 4_800_000.0)
    gap_years_ci80: tuple = (0.05, 2.5)   # cost/speed/deployment gap after the threshold is crossed
    # stage 2: takeoff (AI-2027 takeoff forecast)
    p_sc_is_sar: float = 0.15
    w_sc_sar_ci80: tuple = (1.5, 10.0)    # human-only years, lognormal
    y_med_sar_ci80: tuple = (1.0, 25.0)   # automated-median -> SAR human-only years (for SAR->SIAR)
    corr_sc_sar: float = 0.8
    # SIAR -> ASI: AI-2027 gives median 95 years, 80% CI 2.4 to 1,000,000, and calls the upper end
    # "probably too high". We keep their median and lower bound and make the interval symmetric
    # in log space (upper ~3,800 years).
    w_siar_asi_median: float = 95.0
    w_siar_asi_lo80: float = 2.4
    m_sc_ci90: tuple = (3.0, 10.0)        # AI R&D progress multiplier at AC (AI-2027: 5; 5.8 by Guesstimate)
    m_sar_ci90: tuple = (8.0, 80.0)       # at SAR (AI-2027: 25; methods 2-3 gave 8-139, 5-112)
    m_siar_ratio_ci90: tuple = (3.0, 30.0)  # SIAR / SAR (AI-2027: ~10x, 250)
    m_asi_ratio_ci90: tuple = (3.0, 20.0)   # ASI / SIAR (AI-2027: 8x, 2000)


@dataclass
class Posterior:
    y_a: np.ndarray
    D_a: np.ndarray
    g: np.ndarray
    regime: np.ndarray   # index into REGIMES
    sigma_x: np.ndarray
    w: np.ndarray        # normalized importance weights
    ess: float
    regime_post: dict
    log_evidence: float


def log2h_at(t, y_a, D_a, g):
    """Frontier log2 horizon at decimal year t (vectorized over samples). Returns +inf past a
    superexponential blow-up and -inf before a subexponential trend's start."""
    tau = np.asarray(t) - T_ANCHOR
    out = np.empty(np.broadcast(tau, y_a, D_a, g).shape)
    g = np.broadcast_to(g, out.shape)
    tau = np.broadcast_to(tau, out.shape)
    y_a = np.broadcast_to(y_a, out.shape)
    D_a = np.broadcast_to(D_a, out.shape)
    flat = np.abs(g - 1.0) < 1e-9
    out[flat] = y_a[flat] + tau[flat] / D_a[flat]
    nf = ~flat
    arg = 1.0 - (1.0 - g[nf]) * tau[nf] / D_a[nf]
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.log(arg) / np.log(g[nf])
    bad = arg <= 0
    n[bad & (g[nf] < 1)] = np.inf
    n[bad & (g[nf] > 1)] = -np.inf
    out[nf] = y_a[nf] + n
    return out


def time_to_doublings(n, D_a, g):
    """Years after T_ANCHOR at which n more doublings (relative to y_a) are reached."""
    n = np.asarray(n, dtype=float)
    flat = np.abs(g - 1.0) < 1e-9
    with np.errstate(over="ignore", invalid="ignore"):
        tau = np.where(flat, D_a * n, D_a * (1.0 - np.power(g, n)) / np.where(flat, 1.0, 1.0 - g))
    return tau


def sample_prior(n: int, pr: Priors, rng: np.random.Generator) -> dict:
    regime = rng.choice(3, size=n, p=[pr.regime_prior[r] for r in REGIMES])
    g = np.ones(n)
    mu, s = ln_from_ci(*pr.g_super_ci80)
    k = regime == 0
    g[k] = np.minimum(np.exp(rng.normal(mu, s, k.sum())), 0.999)
    mu, s = ln_from_ci(*pr.g_sub_ci80)
    k = regime == 2
    g[k] = np.maximum(np.exp(rng.normal(mu, s, k.sum())), 1.001)
    mu, s = ln_from_ci(*pr.D_a_ci80)
    return {
        "y_a": rng.normal(pr.y_a_mean, pr.y_a_sd, n),
        "D_a": np.exp(rng.normal(mu, s, n)),
        "g": g, "regime": regime,
        "sigma_x": np.abs(rng.normal(0, pr.sigma_x_scale, n)),
    }


def student_t_logpdf(x, scale, df):
    from scipy.special import gammaln
    z = x / scale
    return (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
            - (df + 1) / 2 * np.log1p(z * z / df))


def _log_prior(p: dict, pr: Priors) -> np.ndarray:
    from scipy.stats import norm
    lp = norm.logpdf(p["y_a"], pr.y_a_mean, pr.y_a_sd)
    mu, s = ln_from_ci(*pr.D_a_ci80)
    lp += norm.logpdf(np.log(p["D_a"]), mu, s) - np.log(p["D_a"])
    lp += np.log(2) + norm.logpdf(p["sigma_x"], 0, pr.sigma_x_scale)
    lp += np.log(np.array([pr.regime_prior[r] for r in REGIMES]))[p["regime"]]
    for k, ci in ((0, pr.g_super_ci80), (2, pr.g_sub_ci80)):
        sel = p["regime"] == k
        mu, s = ln_from_ci(*ci)
        lp[sel] += norm.logpdf(np.log(p["g"][sel]), mu, s) - np.log(p["g"][sel])
    return lp


def _proposal_from(p: dict, w: np.ndarray, n: int, rng, inflate: float = 2.0, floor: float = 0.05):
    """Gaussian proposal per regime fitted to weighted samples (population Monte Carlo step).
    Returns samples and their log proposal density."""
    from scipy.stats import norm, multivariate_normal
    rw = np.array([w[p["regime"] == k].sum() for k in range(3)])
    rw = (rw + floor) / (rw + floor).sum()
    regime = rng.choice(3, size=n, p=rw)
    out = {"y_a": np.empty(n), "D_a": np.empty(n), "g": np.ones(n), "sigma_x": np.empty(n), "regime": regime}
    logq = np.log(rw)[regime]
    for k in range(3):
        sel_old = p["regime"] == k
        sel = regime == k
        if not sel.any():
            continue
        cols = [p["y_a"][sel_old], np.log(p["D_a"][sel_old]), np.log(p["sigma_x"][sel_old] + 1e-6)]
        if k != 1:
            cols.append(np.log(p["g"][sel_old]))
        X = np.stack(cols, axis=1)
        ww = w[sel_old]
        if ww.sum() <= 0 or sel_old.sum() < 10:
            ww = np.ones(sel_old.sum())
        ww = ww / ww.sum()
        mean = ww @ X
        cov = (X - mean).T @ ((X - mean) * ww[:, None]) * inflate ** 2 + np.eye(X.shape[1]) * 1e-4
        Z = rng.multivariate_normal(mean, cov, size=sel.sum())
        logq[sel] += multivariate_normal(mean, cov, allow_singular=True).logpdf(Z)
        out["y_a"][sel] = Z[:, 0]
        out["D_a"][sel] = np.exp(Z[:, 1])
        sx = np.exp(Z[:, 2])
        out["sigma_x"][sel] = sx
        logq[sel] -= Z[:, 1] + Z[:, 2]          # Jacobians of the log transforms
        if k != 1:
            gg = np.exp(Z[:, 3])
            out["g"][sel] = gg
            logq[sel] -= Z[:, 3]
    # keep each regime's g on its own side of 1, as the prior does
    bad = ((regime == 0) & (out["g"] >= 1)) | ((regime == 2) & (out["g"] <= 1))
    return out, logq, bad


def _loglik(obs: Obs, p: dict, pr: Priors) -> np.ndarray:
    n = len(p["y_a"])
    ll = np.zeros(n)
    for t, y, se in zip(obs.t, obs.y, obs.se):
        mu = log2h_at(t, p["y_a"], p["D_a"], p["g"])
        scale = np.sqrt(se ** 2 + p["sigma_x"] ** 2)
        with np.errstate(invalid="ignore"):
            l = student_t_logpdf(y - mu, scale, pr.t_df)
        l[~np.isfinite(mu)] = -np.inf
        ll += l
    return ll


def fit(obs: Obs, pr: Priors = Priors(), n: int = 200_000, seed: int = 0,
        log_lr_by_regime: np.ndarray | None = None, pmc_rounds: int = 2) -> Posterior:
    """Posterior by importance sampling: one pass from the prior, then `pmc_rounds` passes from a
    Gaussian proposal fitted to the previous weighted sample (population Monte Carlo). The final
    weights are prior x likelihood / proposal, so the result is still exact in the limit.
    log_lr_by_regime (len 3) adds capability-event evidence."""
    rng = np.random.default_rng(seed)
    lr = np.zeros(3) if log_lr_by_regime is None else np.asarray(log_lr_by_regime)
    p = sample_prior(n, pr, rng)
    ll = _loglik(obs, p, pr) + lr[p["regime"]]
    lw = ll
    log_evidence = None
    for _ in range(pmc_rounds + 1):
        m = np.max(lw)
        w = np.exp(lw - m)
        log_evidence = float(m + np.log(np.mean(w)))
        w = w / w.sum()
        if _ == pmc_rounds:
            break
        q, logq, bad = _proposal_from(p, w, n, rng)
        lw_new = _log_prior(q, pr) + _loglik(obs, q, pr) + lr[q["regime"]] - logq
        lw_new[bad | ~np.isfinite(lw_new)] = -np.inf
        p, lw = q, lw_new
    ess = float(1.0 / np.sum(w ** 2))
    regime_post = {r: float(w[p["regime"] == i].sum()) for i, r in enumerate(REGIMES)}
    return Posterior(y_a=p["y_a"], D_a=p["D_a"], g=p["g"], regime=p["regime"], sigma_x=p["sigma_x"],
                     w=w, ess=ess, regime_post=regime_post, log_evidence=log_evidence)


def _takeoff_time(W, mA, mB):
    r = (mB - 1.0) / (mA - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = W * (1.0 - np.log(mB / mA) / np.log(r))
    close = np.abs(np.log(r)) < 1e-9
    T = np.where(close, W / mA, T)
    return np.where(W <= 0, 0.0, T)


@dataclass
class Simulation:
    ac: np.ndarray
    sar: np.ndarray
    siar: np.ndarray
    asi: np.ndarray
    regime: np.ndarray
    g: np.ndarray
    D_now: np.ndarray
    y_now: np.ndarray
    thr: np.ndarray


def simulate(post: Posterior, now: float, pr: Priors = Priors(), m: int = 40_000, seed: int = 1) -> Simulation:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post.w), size=m, p=post.w)
    y_a, D_a, g, reg = post.y_a[idx], post.D_a[idx], post.g[idx], post.regime[idx]
    y_now = log2h_at(now, y_a, D_a, g)
    # condition on AC not yet reached: the threshold must exceed today's frontier horizon
    mu, s = ln_from_ci(*pr.thr_minutes_ci80)
    thr = np.empty(m)
    todo = np.arange(m)
    for _ in range(200):
        draw = np.exp(rng.normal(mu, s, todo.size))
        ok = np.log2(draw) > y_now[todo]
        thr[todo[ok]] = draw[ok]
        todo = todo[~ok]
        if todo.size == 0:
            break
    thr[todo] = 2.0 ** (y_now[todo] + 0.1)
    n_needed = np.log2(thr) - y_a
    t_cross = T_ANCHOR + time_to_doublings(n_needed, D_a, g)
    t_cross = np.where(np.isfinite(t_cross), t_cross, np.inf)
    mu, s = ln_from_ci(*pr.gap_years_ci80)
    ac = np.maximum(t_cross, now) + np.exp(rng.normal(mu, s, m))

    # takeoff gaps (human-only years), AI-2027 structure
    z1 = rng.normal(size=m)
    z2 = pr.corr_sc_sar * z1 + math.sqrt(1 - pr.corr_sc_sar ** 2) * rng.normal(size=m)
    mu1, s1 = ln_from_ci(*pr.w_sc_sar_ci80)
    W1 = np.where(rng.random(m) < pr.p_sc_is_sar, 0.0, np.exp(mu1 + s1 * z1))
    mu2, s2 = ln_from_ci(*pr.y_med_sar_ci80)
    X = 10.0 + np.exp(mu2 + s2 * z2)
    W2 = X * ((X / 10.0) ** 2 - 1.0)
    mu3 = math.log(pr.w_siar_asi_median)
    s3 = (mu3 - math.log(pr.w_siar_asi_lo80)) / Z80
    W3 = np.exp(rng.normal(mu3, s3, m))
    # progress multipliers, each stage built on the last so they stay ordered
    def ln90(ci):
        return ln_from_ci(*ci, z=Z90)
    a, b = ln90(pr.m_sc_ci90); m_sc = np.maximum(np.exp(rng.normal(a, b, m)), 1.05)  # a multiplier < 1 would mean AI slows research
    a, b = ln90(pr.m_sar_ci90); m_sar = np.maximum(np.exp(rng.normal(a, b, m)), m_sc * 1.05)
    a, b = ln90(pr.m_siar_ratio_ci90); m_siar = m_sar * np.maximum(np.exp(rng.normal(a, b, m)), 1.05)
    a, b = ln90(pr.m_asi_ratio_ci90); m_asi = m_siar * np.maximum(np.exp(rng.normal(a, b, m)), 1.05)
    sar = ac + _takeoff_time(W1, m_sc, m_sar)
    siar = sar + _takeoff_time(W2, m_sar, m_siar)
    asi = siar + _takeoff_time(W3, m_siar, m_asi)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_now = np.where(np.abs(g - 1) < 1e-9, D_a, D_a * np.power(g, y_now - y_a))
    return Simulation(ac=ac, sar=sar, siar=siar, asi=asi, regime=reg, g=g, D_now=D_now, y_now=y_now, thr=thr)


# ---------------------------------------------------------------- capability events
EVENTS = Path(__file__).resolve().parent.parent / "events" / "events.yaml"


def event_log_lr(cutoff: float | None = None, path: Path = EVENTS) -> tuple[np.ndarray, list]:
    """Sum of log likelihood ratios over regimes from accepted events dated <= cutoff.
    Each event's log LR is scaled by its `confidence` (0..1) and capped at |ln 3| per regime."""
    if not path.exists():
        return np.zeros(3), []
    from .data import year_frac
    doc = yaml.safe_load(path.read_text()) or {}
    total = np.zeros(3)
    used = []
    for ev in doc.get("events", []):
        if ev.get("status") not in ("accepted",):
            continue
        t = year_frac(ev["date"])
        if cutoff is not None and t > cutoff:
            continue
        lr = ev["likelihood_ratio"]
        v = np.array([math.log(lr["super"]), math.log(lr["exp"]), math.log(lr["sub"])])
        v = np.clip(v, -math.log(3), math.log(3)) * float(ev.get("confidence", 1.0))
        total += v
        used.append({"id": ev["id"], "date": str(ev["date"]), "title": ev["title"], "log_lr": v.round(3).tolist()})
    return total, used
