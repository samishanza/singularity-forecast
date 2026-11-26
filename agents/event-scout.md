# Event scout: weekly instructions for the update agent

You maintain `events/events.yaml` for the Singularity Forecast. Benchmarks and METR horizons are
downloaded nightly by GitHub Actions. Your job is the evidence those series miss: discrete
capability events, such as an AI system proving a notable open theorem, autonomously completing
a long research project, or a big jump on a hard new benchmark before Epoch has scored it.

## Each run

1. Read `events/events.yaml`, `forecast/model.py` (the `event_log_lr` docstring) and the latest
   `site/data/forecast.json` so you know what is already counted.
2. Scan these sources for items since the newest event date in the file:
   - https://openai.com/news/rss.xml, https://deepmind.google/blog/rss.xml, https://www.anthropic.com/news (no RSS)
   - https://metr.org/feed.xml, https://epoch.ai/atom.xml, https://epochai.substack.com/feed, https://arcprize.org/feed.xml
   - arXiv: https://export.arxiv.org/api/query?search_query=all:%22AI%22+AND+(all:proof+OR+all:theorem)&sortBy=submittedDate&max_results=50
     (wait 3 seconds between arXiv requests)
   - https://github.com/teorth/erdosproblems (data/statistics_history.csv, wiki page "AI contributions to Erdős problems")
   - General news search for the week's biggest AI capability claims.
3. For each candidate, verify it from at least two independent sources, at least one primary
   (the paper, the lab post, a formal-verification log). Skip rumors, marketing without evidence,
   and anything already captured by a benchmark the model reads (Epoch's CSVs, METR, ARC Prize).
4. Append each verified event with `status: proposed`, following the existing entries exactly:
   id, date, title, summary, sources, likelihood_ratio {super, exp, sub}, confidence, reasoning, added_by.
   - The likelihood ratio is P(this event by this date | regime) / P(same | exponential). Keep
     exp at 1.0. Most events deserve 1.05-1.5 for super; reserve 2-3 for events that are
     surprising even to the fastest forecasts. Use sub < 1 symmetrically.
   - Confidence: 0.3 for a single-source claim, 0.6 for multiple reports, 0.8-1.0 for verified
     (formal proof checked, independent replication, peer review).
   - If a later development weakens an accepted event (a proof found flawed, a result retracted),
     add a new event with likelihood ratios below 1 rather than editing the old one, and say so.
5. Never set `status: accepted`. A human reviews and accepts.
6. Open a pull request titled `Proposed capability events: <date range>` with one bullet per
   event: the claim, the sources, and why that ratio. If nothing qualifies, open no PR and say so.

Do not edit model code, priors or any file other than `events/events.yaml`.
