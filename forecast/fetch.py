"""Download the open AI-progress datasets the model uses.

Every source is public and needs no API key. Raw files go to data/raw/ (not committed);
fetch_log.json records when each was fetched, its size, and its ETag/Last-Modified so a
scheduled run can tell whether anything changed.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = "Mozilla/5.0 (compatible; singularity-forecast/0.1)"

SOURCES = {
    # METR 50%/80% task time horizons per model, with CIs (frontier models, release dates)
    "metr_horizons.yaml": "https://metr.org/assets/benchmark_results_1_1.yaml",
    # Epoch Capabilities Index: one IRT-style latent capability score per model, fresh daily
    "eci_scores.csv": "https://epoch.ai/data/eci_scores.csv",
    # Epoch benchmarking hub: ~90 benchmark CSVs (GPQA, FrontierMath, SWE-bench, HLE, ARC-AGI, ...)
    "benchmark_data.zip": "https://epoch.ai/data/benchmark_data.zip",
    # Epoch notable models: training compute and release dates
    "notable_ai_models.csv": "https://epoch.ai/data/notable_ai_models.csv",
    # Epoch AI supercomputers: H100-equivalents over time (compute build-out)
    "ai_supercomputers.csv": "https://epoch.ai/data/ai_supercomputers.csv",
    # ARC Prize evaluations (ARC-AGI-1/2/3) and model release dates
    "arc_evaluations.json": "https://arcprize.org/media/data/evaluations.json",
    "arc_models.json": "https://arcprize.org/media/data/models.json",
    # Erdos problems database: solved/open counts over time (a math-research capability signal)
    "erdos_stats_history.csv": "https://raw.githubusercontent.com/teorth/erdosproblems/main/data/statistics_history.csv",
}

# Only these benchmark CSVs are kept from Epoch's zip, to keep the repo small.
KEEP_FROM_ZIP = {
    "gpqa_diamond.csv", "frontiermath.csv", "frontiermath_tiers_1_3_v2.csv", "frontiermath_tier_4_v2.csv",
    "frontiermath_erdos.csv", "swe_bench_verified.csv", "hle_external.csv", "arc_agi_2_external.csv",
    "os_world_external.csv", "osworld_2_external.csv", "metr_time_horizons_external.csv",
    "terminalbench.csv", "gso.csv", "rli.csv", "gdpval.csv", "benchmark_metadata.csv",
}


def _get(url: str) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), {k: r.headers.get(k) for k in ("ETag", "Last-Modified")}


def fetch(only: set[str] | None = None) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    log_path = RAW / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for name, url in SOURCES.items():
        if only and name not in only:
            continue
        try:
            body, headers = _get(url)
        except Exception as e:  # a dead source must not kill the run; the model uses what it has
            log[name] = {**log.get(name, {}), "url": url, "error": str(e),
                         "tried": datetime.now(timezone.utc).isoformat()}
            print(f"FAILED {name}: {e}", file=sys.stderr)
            continue
        if name.endswith(".zip"):
            out = RAW / "epoch_benchmarks"
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                for member in z.namelist():
                    base = member.rsplit("/", 1)[-1]
                    if base in KEEP_FROM_ZIP:
                        (out / base).write_bytes(z.read(member))
        else:
            (RAW / name).write_bytes(body)
        log[name] = {"url": url, "bytes": len(body), "fetched": datetime.now(timezone.utc).isoformat(), **headers}
        print(f"ok {name} ({len(body):,} bytes)")
        time.sleep(1)
    log_path.write_text(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    fetch(set(sys.argv[1:]) or None)
