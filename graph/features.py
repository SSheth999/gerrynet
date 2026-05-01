import os
import math
import geopandas as gpd
import pandas as pd
from census import Census
from dotenv import load_dotenv

load_dotenv()
c = Census(os.getenv("CENSUS_API_KEY"))

ACS_VARS = (
    "B01003_001E",  # total population
    "B02001_002E",  # white alone
    "B02001_003E",  # Black alone
    "B02001_004E",  # Asian alone
    "B03001_003E",  # Hispanic or Latino
    "B19013_001E",  # median household income
)

CYCLE_TO_ACS_YEAR = {
    2012: 2012,
    2016: 2016,
    2020: 2019,
    2022: 2022,
    2024: 2022,
}

CYCLE_TO_ELECTION_YEARS = {
    2012: [2012, 2014, 2016],
    2016: [2016, 2018, 2020],
    2020: [2018, 2020, 2022],
    2022: [2022, 2024, None],
    2024: [2024, None, None],
}

CD_FIELD = {
    2012: "CD113FP",
    2016: "CD115FP",
    2020: "CD116FP",
    2022: "CD118FP",
    2024: "CD119FP",
}

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}

FIPS_TO_STATE: dict[str, str] = {v: k for k, v in STATE_FIPS.items()}


def _compute_polsby_popper(gdf: gpd.GeoDataFrame) -> pd.Series:
    gdf_proj = gdf.to_crs("EPSG:5070")
    area = gdf_proj.geometry.area
    perimeter = gdf_proj.geometry.length
    return (4 * math.pi * area) / (perimeter ** 2)


def _fetch_acs(state_fips: str, acs_year: int) -> pd.DataFrame:
    data = c.acs5.get(
        ACS_VARS,
        {"for": "congressional district:*", "in": f"state:{state_fips}"},
        year=acs_year
    )
    acs = pd.DataFrame(data)
    acs = acs.rename(columns={
        "B01003_001E": "total_pop",
        "B02001_002E": "white_pop",
        "B02001_003E": "black_pop",
        "B02001_004E": "asian_pop",
        "B03001_003E": "hispanic_pop",
        "B19013_001E": "median_income",
    })

    numeric = ["total_pop", "white_pop", "black_pop", "asian_pop", "hispanic_pop", "median_income"]
    acs[numeric] = acs[numeric].apply(pd.to_numeric, errors="coerce")

    acs["pct_black"]    = acs["black_pop"]    / acs["total_pop"]
    acs["pct_asian"]    = acs["asian_pop"]    / acs["total_pop"]
    acs["pct_hispanic"] = acs["hispanic_pop"] / acs["total_pop"]
    acs["pct_nonwhite"] = 1 - (acs["white_pop"] / acs["total_pop"])

    ideal_pop = acs["total_pop"].sum() / len(acs)
    acs["pop_deviation"] = (acs["total_pop"] - ideal_pop) / ideal_pop

    acs["geoid"] = acs["state"].str.zfill(2) + acs["congressional district"].str.zfill(2)

    return acs[[
        "geoid", "total_pop", "pct_black", "pct_asian",
        "pct_hispanic", "pct_nonwhite", "median_income", "pop_deviation"
    ]]


def _fetch_election_margins(state_fips: str, cycle: int) -> pd.DataFrame:
    election_years = CYCLE_TO_ELECTION_YEARS[cycle]
    state_abbr = FIPS_TO_STATE[state_fips]

    # The MIT Election Lab file uses a `.tab` extension but is actually
    # comma-delimited; uppercase "GEN" denotes the general election.
    df = pd.read_csv(
        "data/raw/election/1976-2024-house.tab",
        low_memory=False,
    )
    df = df[(df["stage"] == "GEN") & (df["state_po"] == state_abbr)]

    margins = {}
    for i, year in enumerate(election_years):
        if year is None:
            margins[f"margin_t{i}"] = 0.0
            continue

        year_df = df[df["year"] == year]
        dem = year_df[year_df["party"] == "DEMOCRAT"].groupby("district")["candidatevotes"].sum()
        rep = year_df[year_df["party"] == "REPUBLICAN"].groupby("district")["candidatevotes"].sum()

        combined = pd.concat([dem, rep], axis=1, keys=["dem", "rep"]).fillna(0)
        total = combined["dem"] + combined["rep"]
        combined[f"margin_t{i}"] = (combined["dem"] - combined["rep"]) / total.replace(0, float("nan"))
        margins[f"margin_t{i}"] = combined[f"margin_t{i}"]

    result = pd.DataFrame(margins).reset_index()
    result = result.rename(columns={"district": "district_num"})
    result["geoid"] = state_fips.zfill(2) + result["district_num"].astype(str).str.zfill(2)

    return result[["geoid", "margin_t0", "margin_t1", "margin_t2"]]


def _compute_incumbency(state_fips: str, cycle: int) -> pd.DataFrame:
    state_abbr = FIPS_TO_STATE[state_fips]
    election_years = CYCLE_TO_ELECTION_YEARS[cycle]

    if election_years[0] is None or len(election_years) < 2:
        return pd.DataFrame(columns=["geoid", "incumbency"])

    df = pd.read_csv(
        "data/raw/election/1976-2024-house.tab",
        low_memory=False,
    )
    df = df[(df["stage"] == "GEN") & (df["state_po"] == state_abbr)]

    def winning_party(year):
        year_df = df[df["year"] == year]
        if year_df.empty:
            return pd.Series(dtype=object, name="party")
        idx = year_df.groupby("district")["candidatevotes"].idxmax()
        winners = year_df.loc[idx][["district", "party"]].set_index("district")
        return winners["party"]

    current = winning_party(election_years[0])
    if current.empty:
        return pd.DataFrame(columns=["geoid", "incumbency"])

    previous_year = election_years[0] - 2
    try:
        previous = winning_party(previous_year)
    except Exception:
        previous = pd.Series(dtype=object, name="party")

    # Align previous to current so that districts present this cycle but not
    # in the prior election (e.g. after apportionment changes) compare against
    # NaN -> not equal, yielding incumbency=0 for newly created districts.
    previous_aligned = previous.reindex(current.index)
    same = (current == previous_aligned).astype(int)

    incumbency = same.reset_index()
    incumbency.columns = ["district_num", "incumbency"]
    incumbency["geoid"] = state_fips.zfill(2) + incumbency["district_num"].astype(str).str.zfill(2)
    return incumbency[["geoid", "incumbency"]]


def build_features(gdf: gpd.GeoDataFrame, state_fips: str, cycle: int) -> gpd.GeoDataFrame:
    """Build full 9-feature node feature matrix for one state and cycle.

    Args:
        gdf:        GeoDataFrame for one state, already filtered and normalized.
        state_fips: Two-digit state FIPS code e.g. '42' for Pennsylvania.
        cycle:      Redistricting cycle year: 2012, 2016, 2020, 2022, or 2024.

    Returns:
        GeoDataFrame with all node features attached as columns.
    """
    gdf = gdf.reset_index(drop=True).to_crs("EPSG:4326")

    # build geoid join key
    cd_field = CD_FIELD[cycle]
    gdf["geoid"] = gdf["STATEFP"].str.zfill(2) + gdf[cd_field].str.zfill(2)

    # polsby-popper
    gdf["polsby_popper"] = _compute_polsby_popper(gdf).values

    # ACS demographics
    acs_year = CYCLE_TO_ACS_YEAR[cycle]
    acs = _fetch_acs(state_fips, acs_year)
    gdf = gdf.merge(acs, on="geoid", how="left")

    # election margins
    margins = _fetch_election_margins(state_fips, cycle)
    gdf = gdf.merge(margins, on="geoid", how="left")

    # incumbency
    incumbency = _compute_incumbency(state_fips, cycle)
    gdf = gdf.merge(incumbency, on="geoid", how="left")

    # fill any remaining NaN margins with 0
    for col in ["margin_t0", "margin_t1", "margin_t2", "incumbency"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0)

    return gdf