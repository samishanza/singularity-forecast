# Singularity Forecast

A continuously updated probability distribution for when AI recursive self-improvement reaches
artificial superintelligence (ASI), built on open AI-progress data and the AI-2027 takeoff model.

**How it updates**
- **Nightly (GitHub Actions):** downloads METR time horizons, Epoch AI's capabilities index and
  benchmark hub, ARC Prize results and the Erdős problems database. It then refits the model and
  commits the new `site/data/*.json`, which Vercel redeploys.
- **Weekly (Claude agent):** scans lab blogs, arXiv and math databases for capability events the
  benchmarks miss. It proposes them in `events/events.yaml` as likelihood ratios with sources, and
  opens a pull request. Events count only after a human sets `status: accepted`.

## The model in one paragraph

Fit the frontier METR 50% time horizon with a doubling time that can shrink, stay fixed or grow:
`log2 h(t) = y0 + ln(1 − (1 − g)(t − 2026)/D)/ln g`. A three-regime prior on g, a Student-t
likelihood and importance sampling refined by population Monte Carlo give the posterior. METR's
lag is filled by mapping Epoch's Capabilities Index to horizons. The automated coder arrives when
the horizon crosses an uncertain threshold, conditioned on not having happened yet, plus a gap.
After that, AI-2027's takeoff gaps (SAR, SIAR, ASI) are crossed with rising, uncertain AI R&D
multipliers. Calendar time per gap has the closed form `T = W(1 − ln(mB/mA)/ln r)`. The full
derivation is on the site's Method section and in `forecast/model.py`.

## Run it

```bash
pip install -r requirements.txt
python -m forecast.fetch                         # download data to data/raw/
python -m forecast.build --history --sensitivity --backtest
python -m pytest -q tests
cd site && python -m http.server 8000            # open http://localhost:8000
```

## Layout

| Path | What |
|---|---|
| `forecast/fetch.py` | downloads the open datasets |
| `forecast/data.py` | frontier observations; ECI → horizon nowcast map |
| `forecast/model.py` | priors, posterior fit, takeoff simulation, event updates |
| `forecast/backtest.py` | out-of-sample scoring of the trend model |
| `forecast/build.py` | runs everything; writes `site/data/` |
| `events/events.yaml` | capability events and their likelihood ratios |
| `agents/event-scout.md` | the weekly agent's instructions |
| `site/` | the static website (no build step) |

## Known limitations

See the site's Method section. In short:
- Time horizons are software tasks.
- Takeoff ignores compute growth.
- Backtests show past fits underpredicted progress.
- The takeoff parameters are expert judgment.

Data: METR; Epoch AI (CC-BY-4.0); ARC Prize; teorth/erdosproblems (Apache-2.0).
