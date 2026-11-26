import math

import numpy as np
from scipy.integrate import quad

from forecast.data import Obs
from forecast.model import (Priors, T_ANCHOR, _takeoff_time, fit, ln_from_ci, log2h_at,
                            simulate, time_to_doublings)


def test_ln_from_ci_roundtrip():
    mu, s = ln_from_ci(2.0, 50.0)
    assert math.isclose(math.exp(mu), 10.0, rel_tol=1e-9)
    assert math.isclose(math.exp(mu + 1.281552 * s), 50.0, rel_tol=1e-6)


def test_trend_inverse_consistency():
    # time_to_doublings is the inverse of log2h_at for every regime
    for g in (0.8, 1.0, 1.15):
        for n in (-3.0, 0.5, 4.0):
            tau = time_to_doublings(np.array(n), np.array(0.4), np.array(g))
            y = log2h_at(T_ANCHOR + tau, np.array(7.0), np.array(0.4), np.array(g))
            assert math.isclose(float(y), 7.0 + n, abs_tol=1e-9)


def test_superexponential_blows_up_in_finite_time():
    D, g = 0.4, 0.8
    t_star = T_ANCHOR + D / (1 - g)
    assert np.isfinite(log2h_at(t_star - 1e-3, 7.0, D, g))
    assert np.isinf(log2h_at(t_star + 1e-3, 7.0, D, g))


def test_takeoff_closed_form_matches_integral():
    # T = integral over the gap of dW / m(f), m(f) = 1 + (mA-1) r^f
    for W, mA, mB in ((4.0, 5.0, 25.0), (95.0, 250.0, 2000.0), (3.0, 7.0, 7.0)):
        r = (mB - 1) / (mA - 1)
        num, _ = quad(lambda f: W / (1 + (mA - 1) * r ** f), 0, 1)
        got = float(_takeoff_time(np.array(W), np.array(mA), np.array(mB)))
        assert math.isclose(got, num, rel_tol=1e-6)
    # AI-2027 reports ~0.16 calendar years for SIAR->ASI at the median
    assert abs(float(_takeoff_time(np.array(95.0), np.array(250.0), np.array(2000.0))) - 0.16) < 0.01


def test_fit_recovers_synthetic_exponential_trend():
    rng = np.random.default_rng(3)
    t = np.linspace(2023.0, 2026.5, 20)
    true_D = 0.35
    y = 8.0 + (t - T_ANCHOR) / true_D + rng.normal(0, 0.3, t.size)
    obs = Obs(t=t, y=y, se=np.full(t.size, 0.3), kind=["metr"] * t.size, name=[str(i) for i in range(t.size)])
    post = fit(obs, Priors(), n=60_000, seed=1)
    o = np.argsort(post.D_a); c = np.cumsum(post.w[o])
    med_D = post.D_a[o][np.searchsorted(c, 0.5)]
    assert abs(med_D - true_D) < 0.05
    assert post.regime_post["exp"] > 0.3
    assert post.ess > 1000


def test_simulation_orders_milestones_and_respects_now():
    t = np.linspace(2023.0, 2026.5, 12)
    obs = Obs(t=t, y=8.0 + (t - T_ANCHOR) / 0.4, se=np.full(t.size, 0.4), kind=["metr"] * 12, name=["x"] * 12)
    post = fit(obs, Priors(), n=30_000, seed=2)
    sim = simulate(post, now=2026.9, m=5000, seed=3)
    assert np.all(sim.ac >= 2026.9)
    assert np.all(sim.sar >= sim.ac) and np.all(sim.siar >= sim.sar) and np.all(sim.asi >= sim.siar)
    assert np.all(np.log2(sim.thr) > sim.y_now)  # conditioned on AC not having happened yet
