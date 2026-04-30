"""Download U.S. Census TIGER/Line Congressional District shapefiles.

The Census FTP layout is not uniform across vintages:
  * 113th–117th Congress shapefiles are published as a single national zip
    (tl_<year>_us_cd<NNN>.zip).
  * 118th and 119th Congress shapefiles are only published per-state
    (tl_<year>_<stateFIPS>_cd<NNN>.zip). There is NO national zip.

This script handles both styles and writes everything under
data/raw/shapefiles/<plan>/...
"""

from __future__ import annotations

import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DEST = Path("data/raw/shapefiles")
TIMEOUT = 120
CHUNK = 1024 * 1024
RETRIES = 3
BACKOFF_SECONDS = 5
USER_AGENT = "gerrynet-data-fetcher/0.1 (+https://example.com)"

# State + DC + PR + Island Area FIPS codes used in the per-state TIGER files.
STATE_FIPS: tuple[str, ...] = (
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56", "60", "66", "69", "72",
    "78",
)


@dataclass(frozen=True)
class Plan:
    name: str          
    vintage: str       
    year: str          
    code: str        
    scope: str  


PLANS: tuple[Plan, ...] = (
    Plan("cd113", "TIGER2013", "2013", "cd113", "national"),
    Plan("cd115", "TIGER2016", "2016", "cd115", "national"),
    Plan("cd116", "TIGER2020", "2020", "cd116", "national"),
    Plan("cd118", "TIGER2023", "2023", "cd118", "state"),
    Plan("cd119", "TIGER2025", "2025", "cd119", "state"),
)


def urls_for(plan: Plan) -> list[tuple[str, str]]:
    """Return list of (filename, url) tuples for a plan."""
    base = f"https://www2.census.gov/geo/tiger/{plan.vintage}/CD"
    if plan.scope == "national":
        fname = f"tl_{plan.year}_us_{plan.code}.zip"
        return [(fname, f"{base}/{fname}")]
    return [
        (
            f"tl_{plan.year}_{fips}_{plan.code}.zip",
            f"{base}/tl_{plan.year}_{fips}_{plan.code}.zip",
        )
        for fips in STATE_FIPS
    ]


def download(url: str, out_path: Path, session: requests.Session) -> bool:
    """Download with retries. Returns True on success, False on permanent failure."""
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    print(f"  404 (skipping): {url}")
                    return False
                r.raise_for_status()
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                return True
        except requests.RequestException as e:
            wait = BACKOFF_SECONDS * attempt
            print(f"  attempt {attempt}/{RETRIES} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    print(f"  giving up: {url}")
    return False


def extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=dest)


def fetch_plan(plan: Plan, session: requests.Session) -> None:
    plan_dir = DEST / plan.name
    plan_dir.mkdir(parents=True, exist_ok=True)
    targets = urls_for(plan)
    print(f"\n=== {plan.name} ({plan.scope}, {len(targets)} file(s)) ===")

    for fname, url in targets:
        zip_path = plan_dir / fname
        marker = plan_dir / (fname + ".extracted")
        if marker.exists():
            continue
        if not zip_path.exists():
            print(f"  downloading {fname}")
            if not download(url, zip_path, session):
                continue
        else:
            print(f"  using cached {fname}")
        try:
            extract(zip_path, plan_dir)
            marker.touch()
        except zipfile.BadZipFile:
            print(f"  bad zip, removing: {zip_path}")
            zip_path.unlink(missing_ok=True)


def main(argv: Iterable[str]) -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    wanted = set(argv)
    plans = [p for p in PLANS if not wanted or p.name in wanted]
    if wanted and not plans:
        print(f"No matching plans for {sorted(wanted)}. Known: {[p.name for p in PLANS]}")
        return 2

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for plan in plans:
            fetch_plan(plan, session)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
