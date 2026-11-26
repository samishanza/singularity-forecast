# Security

- No credentials live in this repo. The nightly workflow reads one secret, `VERCEL_TOKEN`, a
  repository secret scoped to the project's Vercel team.
- The workflow runs only on schedule, manual dispatch, and pushes to `main`. It never runs on pull
  requests, so code from forks never sees the secret. Third-party actions are pinned to commit SHAs,
  and the Vercel CLI to an exact version.
- Downloaded datasets are parsed as data only: `yaml.safe_load`, CSV and JSON, zip members
  extracted by basename to a fixed folder. The site inserts all text with `textContent` and is
  served with a strict Content-Security-Policy.
- Capability events proposed by the weekly agent come from the open web. They only take effect
  after a human reviews the pull request and sets `status: accepted`.

To report a problem, open a private security advisory on GitHub.
