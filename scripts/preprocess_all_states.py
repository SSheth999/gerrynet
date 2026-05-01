"""Preprocess every (state, cycle) pair into node features + adjacency edges.

For each U.S. state and each redistricting cycle (2012, 2016, 2020, 2022, 2024)
this script:

1. Loads the appropriate Census TIGER/Line shapefile.
2. Filters to the state and drops non-voting / placeholder district codes.
3. Calls ``build_features`` to attach demographics, election margins, etc.
4. Calls ``adjacency`` to compute the inter-district edge list.
5. Writes ``data/processed/{state}_{cycle}.geojson`` and
   ``data/processed/{state}_{cycle}_edges.csv``.

(state, cycle) pairs whose outputs already exist on disk are skipped, so the
script is safe to re-run after partial failures.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import geopandas as gpd

# Allow ``python scripts/preprocess_all_states.py`` to import the ``graph``
# package even when the project root is not on PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.adjacency import adjacency
from graph.features import STATE_FIPS, build_features


SHAPEFILE_ROOT = Path("data/raw/shapefiles")
PROCESSED_DIR = Path("data/processed")

# Each cycle maps to (plan_dir, tiger_year, scope, cd_field).
#   plan_dir   : subdirectory under data/raw/shapefiles/
#   tiger_year : the year embedded in the TIGER filename
#   scope      : "national" -> one tl_<year>_us_<plan>.shp covering all states
#                "state"    -> one tl_<year>_<fips>_<plan>.shp per state
#   cd_field   : column in the shapefile that holds the district number
CYCLE_TO_PLAN: dict[int, tuple[str, str, str, str]] = {
    2012: ("cd113", "2013", "national", "CD113FP"),
    2016: ("cd115", "2016", "national", "CD115FP"),
    2020: ("cd116", "2020", "national", "CD116FP"),
    2022: ("cd118", "2023", "state",    "CD118FP"),
    2024: ("cd119", "2025", "state",    "CD119FP"),
}

# District codes that don't represent a real, voting congressional district.
# "ZZ" = "no congressional district at this time" (Census sentinel),
# "98" / "99" = non-voting delegate / unassigned territories.
BAD_CD_CODES = {"ZZ", "98", "99"}


def _shapefile_path(state_fips: str, cycle: int) -> Path:
    plan, year, scope, _ = CYCLE_TO_PLAN[cycle]
    if scope == "national":
        return SHAPEFILE_ROOT / plan / f"tl_{year}_us_{plan}.shp"
    return SHAPEFILE_ROOT / plan / f"tl_{year}_{state_fips}_{plan}.shp"


def load_state_gdf(state_fips: str, cycle: int) -> gpd.GeoDataFrame:
    """Load the right shapefile for ``cycle``, filter to one state, return it.

    Drops rows whose district code is a non-voting placeholder ("ZZ", "98", "99").
    The returned GeoDataFrame has its index reset and is in the shapefile's
    native CRS (typically EPSG:4269); ``build_features`` and ``adjacency`` will
    reproject as needed.
    """
    plan, _, scope, cd_field = CYCLE_TO_PLAN[cycle]
    shp = _shapefile_path(state_fips, cycle)
    if not shp.exists():
        raise FileNotFoundError(f"Missing shapefile for cycle {cycle}: {shp}")

    gdf = gpd.read_file(shp)

    # National files contain every state; per-state files are already filtered
    # but we still apply the predicate defensively.
    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"].astype(str).str.zfill(2) == state_fips]

    if cd_field not in gdf.columns:
        raise KeyError(
            f"Expected district column '{cd_field}' missing from {shp.name}; "
            f"got columns: {list(gdf.columns)}"
        )

    gdf = gdf[~gdf[cd_field].astype(str).isin(BAD_CD_CODES)]
    gdf = gdf.reset_index(drop=True)

    if "STATEFP" in gdf.columns:
        gdf["STATEFP"] = gdf["STATEFP"].astype(str).str.zfill(2)
    gdf[cd_field] = gdf[cd_field].astype(str).str.zfill(2)

    return gdf


def _output_paths(state_abbr: str, cycle: int) -> tuple[Path, Path]:
    geojson = PROCESSED_DIR / f"{state_abbr}_{cycle}.geojson"
    edges = PROCESSED_DIR / f"{state_abbr}_{cycle}_edges.csv"
    return geojson, edges


def process_one(state_abbr: str, state_fips: str, cycle: int) -> str:
    """Process a single (state, cycle). Returns a short status string."""
    geojson_path, edges_path = _output_paths(state_abbr, cycle)
    if geojson_path.exists() and edges_path.exists():
        return "skip (cached)"

    gdf = load_state_gdf(state_fips, cycle)
    if gdf.empty:
        return "skip (no districts)"

    features_gdf = build_features(gdf, state_fips, cycle)
    edges_df = adjacency(features_gdf)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if geojson_path.exists():
        geojson_path.unlink()
    features_gdf.to_file(geojson_path, driver="GeoJSON")
    edges_df.to_csv(edges_path, index=False)

    return f"ok ({len(features_gdf)} districts, {len(edges_df)} edges)"


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    states = sorted(STATE_FIPS.items())  # [(abbr, fips), ...]
    cycles = sorted(CYCLE_TO_PLAN.keys())
    total = len(states) * len(cycles)

    failures: list[tuple[str, int, str]] = []
    i = 0
    for cycle in cycles:
        for abbr, fips in states:
            i += 1
            label = f"[{i:>3}/{total}] {abbr} {cycle}"
            try:
                status = process_one(abbr, fips, cycle)
                print(f"{label}: {status}")
            except Exception as exc:  # noqa: BLE001 - we want the loop to continue
                failures.append((abbr, cycle, str(exc)))
                print(f"{label}: FAIL - {exc}")
                traceback.print_exc()

    if failures:
        print(f"\n{len(failures)} failures:")
        for abbr, cycle, msg in failures:
            print(f"  {abbr} {cycle}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
