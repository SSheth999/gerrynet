"""Unit tests for ``graph.features``.

External I/O (the Census API and the on-disk election ``.tab`` file) is
mocked, so the suite is hermetic and runs in well under a second.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from graph import features
from graph.features import (
    ACS_VARS,
    CD_FIELD,
    CYCLE_TO_ACS_YEAR,
    CYCLE_TO_ELECTION_YEARS,
    FIPS_TO_STATE,
    STATE_FIPS,
    _compute_incumbency,
    _compute_polsby_popper,
    _fetch_acs,
    _fetch_election_margins,
    build_features,
)

ELECTION_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "election" / "1976-2024-house.tab"

# ---------------------------------------------------------------------------
# Constants / lookup tables
# ---------------------------------------------------------------------------


class TestConstants:
    def test_state_fips_size_50(self):
        assert len(STATE_FIPS) == 50

    def test_state_fips_values_are_two_digit_strings(self):
        for code in STATE_FIPS.values():
            assert isinstance(code, str)
            assert len(code) == 2 and code.isdigit()

    def test_fips_to_state_is_inverse_of_state_fips(self):
        for abbr, fips in STATE_FIPS.items():
            assert FIPS_TO_STATE[fips] == abbr
        assert len(FIPS_TO_STATE) == len(STATE_FIPS)

    def test_cycle_keys_consistent_across_lookup_tables(self):
        cycles = set(CYCLE_TO_ACS_YEAR)
        assert cycles == set(CYCLE_TO_ELECTION_YEARS)
        assert cycles == set(CD_FIELD)

    def test_cycle_to_election_years_has_three_slots(self):
        for cycle, years in CYCLE_TO_ELECTION_YEARS.items():
            assert len(years) == 3, f"cycle {cycle} has {len(years)} years"

    def test_acs_vars_has_six_entries(self):
        assert len(ACS_VARS) == 6
        assert all(isinstance(v, str) for v in ACS_VARS)

    def test_cd_field_is_well_formed(self):
        for cycle, field in CD_FIELD.items():
            assert field.startswith("CD")
            assert field.endswith("FP")


# ---------------------------------------------------------------------------
# _compute_polsby_popper
# ---------------------------------------------------------------------------


def _gdf_from_geoms(geoms):
    # Pass EPSG:5070 directly so to_crs() inside the function is a no-op.
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:5070")


class TestComputePolsbyPopper:
    def test_circle_close_to_one(self):
        circle = Point(0, 0).buffer(1000, quad_segs=128)
        gdf = _gdf_from_geoms([circle])
        pp = _compute_polsby_popper(gdf)
        assert pp.iloc[0] == pytest.approx(1.0, rel=1e-3)

    def test_square_equals_pi_over_four(self):
        square = box(0, 0, 100, 100)
        gdf = _gdf_from_geoms([square])
        pp = _compute_polsby_popper(gdf)
        assert pp.iloc[0] == pytest.approx(math.pi / 4, rel=1e-9)

    def test_thin_rectangle_is_low(self):
        thin = box(0, 0, 1000, 1)
        gdf = _gdf_from_geoms([thin])
        pp = _compute_polsby_popper(gdf)
        assert pp.iloc[0] < 0.05

    def test_returns_one_value_per_geometry(self):
        geoms = [Point(0, 0).buffer(100), box(0, 0, 100, 100), box(0, 0, 1000, 1)]
        gdf = _gdf_from_geoms(geoms)
        pp = _compute_polsby_popper(gdf)
        assert isinstance(pp, pd.Series)
        assert len(pp) == 3
        assert (pp > 0).all()
        assert (pp <= 1.0 + 1e-9).all()

    def test_does_not_mutate_input_crs(self):
        gdf = _gdf_from_geoms([box(0, 0, 100, 100)])
        original_crs = gdf.crs
        _compute_polsby_popper(gdf)
        assert gdf.crs == original_crs


# ---------------------------------------------------------------------------
# _fetch_acs
# ---------------------------------------------------------------------------


def _acs_payload(rows):
    """Convert friendly dicts into the list-of-dicts that the Census client
    returns. Numeric values are stringified to mirror the wire format."""
    return [
        {
            "B01003_001E": str(r["total_pop"]),
            "B02001_002E": str(r["white_pop"]),
            "B02001_003E": str(r["black_pop"]),
            "B02001_004E": str(r["asian_pop"]),
            "B03001_003E": str(r["hispanic_pop"]),
            "B19013_001E": str(r["median_income"]),
            "state": r["state"],
            "congressional district": r["cd"],
        }
        for r in rows
    ]


class TestFetchAcs:
    def test_columns_and_geoid_composition(self):
        payload = _acs_payload([
            {"total_pop": 1000, "white_pop": 600, "black_pop": 200,
             "asian_pop": 100, "hispanic_pop": 150, "median_income": 50000,
             "state": "42", "cd": "01"},
            {"total_pop": 1200, "white_pop": 700, "black_pop": 300,
             "asian_pop": 100, "hispanic_pop": 200, "median_income": 60000,
             "state": "42", "cd": "02"},
        ])
        with patch.object(features.c.acs5, "get", return_value=payload) as mocked:
            out = _fetch_acs("42", 2020)

        mocked.assert_called_once()
        # The function passes year as a kwarg.
        _, kwargs = mocked.call_args
        assert kwargs.get("year") == 2020

        assert list(out.columns) == [
            "geoid", "total_pop", "pct_black", "pct_asian",
            "pct_hispanic", "pct_nonwhite", "median_income", "pop_deviation",
        ]
        assert out["geoid"].tolist() == ["4201", "4202"]

    def test_percentage_columns_correct(self):
        payload = _acs_payload([
            {"total_pop": 1000, "white_pop": 600, "black_pop": 200,
             "asian_pop": 100, "hispanic_pop": 150, "median_income": 50000,
             "state": "06", "cd": "07"},
        ])
        with patch.object(features.c.acs5, "get", return_value=payload):
            out = _fetch_acs("06", 2020)

        row = out.iloc[0]
        assert row["pct_black"] == pytest.approx(0.20)
        assert row["pct_asian"] == pytest.approx(0.10)
        assert row["pct_hispanic"] == pytest.approx(0.15)
        assert row["pct_nonwhite"] == pytest.approx(0.40)
        assert row["geoid"] == "0607"

    def test_pop_deviation_centers_on_zero(self):
        payload = _acs_payload([
            {"total_pop": 800, "white_pop": 400, "black_pop": 200,
             "asian_pop": 50, "hispanic_pop": 50, "median_income": 40000,
             "state": "06", "cd": "01"},
            {"total_pop": 1200, "white_pop": 600, "black_pop": 300,
             "asian_pop": 50, "hispanic_pop": 50, "median_income": 60000,
             "state": "06", "cd": "02"},
        ])
        with patch.object(features.c.acs5, "get", return_value=payload):
            out = _fetch_acs("06", 2020)

        # ideal_pop = 1000; deviations should be -0.2 and +0.2.
        assert out["pop_deviation"].sum() == pytest.approx(0.0, abs=1e-12)
        assert sorted(out["pop_deviation"].tolist()) == pytest.approx([-0.2, 0.2])

    def test_non_numeric_income_becomes_nan(self):
        payload = _acs_payload([
            {"total_pop": 1000, "white_pop": 500, "black_pop": 200,
             "asian_pop": 100, "hispanic_pop": 100, "median_income": 0,
             "state": "06", "cd": "01"},
        ])
        # Census uses sentinel strings for "no data"; force the coerce path.
        payload[0]["B19013_001E"] = "N/A"
        with patch.object(features.c.acs5, "get", return_value=payload):
            out = _fetch_acs("06", 2020)
        assert pd.isna(out["median_income"].iloc[0])

    def test_request_includes_state_filter(self):
        payload = _acs_payload([
            {"total_pop": 1, "white_pop": 0, "black_pop": 0, "asian_pop": 0,
             "hispanic_pop": 0, "median_income": 0, "state": "36", "cd": "01"},
        ])
        with patch.object(features.c.acs5, "get", return_value=payload) as mocked:
            _fetch_acs("36", 2022)

        args, _ = mocked.call_args
        # First positional arg is the variables tuple, second is the geo dict.
        assert args[0] == ACS_VARS
        geo = args[1]
        assert geo.get("for") == "congressional district:*"
        assert geo.get("in") == "state:36"


# ---------------------------------------------------------------------------
# _fetch_election_margins
# ---------------------------------------------------------------------------


def _house_df(rows):
    """Build a fake MIT-style U.S. House general-election DataFrame."""
    return pd.DataFrame(
        rows,
        columns=["year", "state_po", "district", "party",
                 "candidatevotes", "stage"],
    )


class TestFetchElectionMargins:
    def test_columns_and_margin_arithmetic(self):
        df = _house_df([
            (2012, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2012, "PA", 1, "REPUBLICAN",  50, "GEN"),
            (2014, "PA", 1, "DEMOCRAT",    80, "GEN"),
            (2014, "PA", 1, "REPUBLICAN",  80, "GEN"),
            (2016, "PA", 1, "DEMOCRAT",    30, "GEN"),
            (2016, "PA", 1, "REPUBLICAN",  70, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _fetch_election_margins("42", 2012)

        assert list(out.columns) == [
            "geoid", "margin_t0", "margin_t1", "margin_t2",
        ]
        assert len(out) == 1
        row = out.iloc[0]
        assert row["geoid"] == "4201"
        assert row["margin_t0"] == pytest.approx((100 - 50) / 150)
        assert row["margin_t1"] == pytest.approx(0.0)
        assert row["margin_t2"] == pytest.approx((30 - 70) / 100)

    def test_none_year_yields_zero_margin(self):
        # Cycle 2024 has [2024, None, None].
        df = _house_df([
            (2024, "PA", 1, "DEMOCRAT",   60, "GEN"),
            (2024, "PA", 1, "REPUBLICAN", 40, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _fetch_election_margins("42", 2024)

        assert out["margin_t0"].iloc[0] == pytest.approx(0.20)
        assert out["margin_t1"].tolist() == [0.0]
        assert out["margin_t2"].tolist() == [0.0]

    def test_filters_to_state_and_general_stage(self):
        df = _house_df([
            (2012, "PA", 1, "DEMOCRAT",       100, "GEN"),
            (2012, "PA", 1, "REPUBLICAN",      50, "GEN"),
            (2012, "OH", 1, "DEMOCRAT",         1, "GEN"),  # other state
            (2012, "OH", 1, "REPUBLICAN",     999, "GEN"),
            (2012, "PA", 1, "DEMOCRAT",    999999, "PRI"),  # primary
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _fetch_election_margins("42", 2012)
        assert out["margin_t0"].iloc[0] == pytest.approx((100 - 50) / 150)

    def test_multiple_districts_get_distinct_geoids(self):
        df = _house_df([
            (2012, "PA", 1, "DEMOCRAT",   60, "GEN"),
            (2012, "PA", 1, "REPUBLICAN", 40, "GEN"),
            (2012, "PA", 2, "DEMOCRAT",   30, "GEN"),
            (2012, "PA", 2, "REPUBLICAN", 70, "GEN"),
            (2014, "PA", 1, "DEMOCRAT",   50, "GEN"),
            (2014, "PA", 1, "REPUBLICAN", 50, "GEN"),
            (2014, "PA", 2, "DEMOCRAT",   55, "GEN"),
            (2014, "PA", 2, "REPUBLICAN", 45, "GEN"),
            (2016, "PA", 1, "DEMOCRAT",   10, "GEN"),
            (2016, "PA", 1, "REPUBLICAN", 90, "GEN"),
            (2016, "PA", 2, "DEMOCRAT",   90, "GEN"),
            (2016, "PA", 2, "REPUBLICAN", 10, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _fetch_election_margins("42", 2012)
        assert sorted(out["geoid"].tolist()) == ["4201", "4202"]


# ---------------------------------------------------------------------------
# _compute_incumbency
# ---------------------------------------------------------------------------


class TestComputeIncumbency:
    def test_same_winning_party_returns_one(self):
        df = _house_df([
            (2012, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2012, "PA", 1, "REPUBLICAN",  90, "GEN"),
            (2010, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2010, "PA", 1, "REPUBLICAN",  50, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _compute_incumbency("42", 2012)

        assert list(out.columns) == ["geoid", "incumbency"]
        assert out.iloc[0]["geoid"] == "4201"
        assert int(out.iloc[0]["incumbency"]) == 1

    def test_different_winning_party_returns_zero(self):
        df = _house_df([
            (2012, "PA", 1, "REPUBLICAN", 100, "GEN"),
            (2012, "PA", 1, "DEMOCRAT",    90, "GEN"),
            (2010, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2010, "PA", 1, "REPUBLICAN",  50, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _compute_incumbency("42", 2012)
        assert int(out.iloc[0]["incumbency"]) == 0

    def test_handles_multiple_districts(self):
        df = _house_df([
            # 2012 winners: D in d1, R in d2
            (2012, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2012, "PA", 1, "REPUBLICAN",  10, "GEN"),
            (2012, "PA", 2, "DEMOCRAT",    10, "GEN"),
            (2012, "PA", 2, "REPUBLICAN", 100, "GEN"),
            # 2010 winners: D in d1 (same), D in d2 (different from R)
            (2010, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2010, "PA", 1, "REPUBLICAN",  10, "GEN"),
            (2010, "PA", 2, "DEMOCRAT",   100, "GEN"),
            (2010, "PA", 2, "REPUBLICAN",  10, "GEN"),
        ])
        with patch("graph.features.pd.read_csv", return_value=df):
            out = _compute_incumbency("42", 2012)
        out = out.set_index("geoid")
        assert int(out.loc["4201", "incumbency"]) == 1
        assert int(out.loc["4202", "incumbency"]) == 0

    def test_returns_empty_when_first_year_is_none(self):
        with patch.dict(features.CYCLE_TO_ELECTION_YEARS,
                        {2012: [None, None, None]}):
            out = _compute_incumbency("42", 2012)
        assert list(out.columns) == ["geoid", "incumbency"]
        assert out.empty

    def test_falls_back_to_zero_when_previous_year_lookup_raises(self):
        df = _house_df([
            (2012, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2012, "PA", 1, "REPUBLICAN",  50, "GEN"),
            (2010, "PA", 1, "DEMOCRAT",   100, "GEN"),
            (2010, "PA", 1, "REPUBLICAN",  50, "GEN"),
        ])
        # Make the *second* call to DataFrame.groupby raise; the first is for
        # the current year (succeeds), the second for the previous year.
        calls = {"n": 0}
        original_groupby = pd.DataFrame.groupby

        def boom_on_second(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated lookup failure")
            return original_groupby(self, *args, **kwargs)

        with patch("graph.features.pd.read_csv", return_value=df), \
             patch.object(pd.DataFrame, "groupby", boom_on_second):
            out = _compute_incumbency("42", 2012)

        assert list(out.columns) == ["geoid", "incumbency"]
        assert out.iloc[0]["geoid"] == "4201"
        assert int(out.iloc[0]["incumbency"]) == 0


# ---------------------------------------------------------------------------
# build_features (end-to-end with the three helpers mocked out)
# ---------------------------------------------------------------------------


def _state_gdf(state_fips, cd_field, cds):
    """Tiny GeoDataFrame mimicking a TIGER/Line CD layer."""
    geoms = [box(i, 0, i + 1, 1) for i in range(len(cds))]
    return gpd.GeoDataFrame(
        {
            "STATEFP": [state_fips] * len(cds),
            cd_field: cds,
            "geometry": geoms,
        },
        crs="EPSG:4326",
    )


class TestBuildFeatures:
    def test_full_set_of_columns_present(self):
        state_fips, cycle = "42", 2012
        cd_field = CD_FIELD[cycle]
        gdf = _state_gdf(state_fips, cd_field, ["01", "02"])

        acs = pd.DataFrame({
            "geoid": ["4201", "4202"],
            "total_pop": [1000, 1200],
            "pct_black": [0.20, 0.25],
            "pct_asian": [0.10, 0.05],
            "pct_hispanic": [0.15, 0.20],
            "pct_nonwhite": [0.40, 0.50],
            "median_income": [50000.0, 60000.0],
            "pop_deviation": [-0.1, 0.1],
        })
        margins = pd.DataFrame({
            "geoid": ["4201", "4202"],
            "margin_t0": [0.10, -0.05],
            "margin_t1": [0.00,  0.00],
            "margin_t2": [-0.20, 0.30],
        })
        incumb = pd.DataFrame({
            "geoid": ["4201", "4202"],
            "incumbency": [1, 0],
        })

        with patch.object(features, "_fetch_acs", return_value=acs), \
             patch.object(features, "_fetch_election_margins", return_value=margins), \
             patch.object(features, "_compute_incumbency", return_value=incumb):
            out = build_features(gdf, state_fips, cycle)

        assert isinstance(out, gpd.GeoDataFrame)
        expected_cols = {
            "geoid", "polsby_popper",
            "total_pop", "pct_black", "pct_asian", "pct_hispanic",
            "pct_nonwhite", "median_income", "pop_deviation",
            "margin_t0", "margin_t1", "margin_t2", "incumbency",
        }
        assert expected_cols.issubset(out.columns)
        assert out["geoid"].tolist() == ["4201", "4202"]

    def test_calls_helpers_with_correct_arguments(self):
        state_fips, cycle = "06", 2020
        cd_field = CD_FIELD[cycle]
        gdf = _state_gdf(state_fips, cd_field, ["01"])

        acs = pd.DataFrame({
            "geoid": ["0601"], "total_pop": [1], "pct_black": [0],
            "pct_asian": [0], "pct_hispanic": [0], "pct_nonwhite": [0],
            "median_income": [0.0], "pop_deviation": [0.0],
        })
        margins = pd.DataFrame({
            "geoid": ["0601"], "margin_t0": [0], "margin_t1": [0], "margin_t2": [0],
        })
        incumb = pd.DataFrame({"geoid": ["0601"], "incumbency": [0]})

        with patch.object(features, "_fetch_acs", return_value=acs) as m_acs, \
             patch.object(features, "_fetch_election_margins", return_value=margins) as m_marg, \
             patch.object(features, "_compute_incumbency", return_value=incumb) as m_inc:
            build_features(gdf, state_fips, cycle)

        m_acs.assert_called_once_with(state_fips, CYCLE_TO_ACS_YEAR[cycle])
        m_marg.assert_called_once_with(state_fips, cycle)
        m_inc.assert_called_once_with(state_fips, cycle)

    def test_missing_margins_and_incumbency_filled_with_zero(self):
        state_fips, cycle = "42", 2012
        cd_field = CD_FIELD[cycle]
        gdf = _state_gdf(state_fips, cd_field, ["01", "02"])

        acs = pd.DataFrame({
            "geoid": ["4201", "4202"],
            "total_pop": [1000, 1200],
            "pct_black": [0, 0], "pct_asian": [0, 0],
            "pct_hispanic": [0, 0], "pct_nonwhite": [0, 0],
            "median_income": [0.0, 0.0], "pop_deviation": [0.0, 0.0],
        })
        # 4202 absent from both margins and incumbency.
        margins = pd.DataFrame({
            "geoid": ["4201"],
            "margin_t0": [0.5], "margin_t1": [0.5], "margin_t2": [0.5],
        })
        incumb = pd.DataFrame({"geoid": ["4201"], "incumbency": [1]})

        with patch.object(features, "_fetch_acs", return_value=acs), \
             patch.object(features, "_fetch_election_margins", return_value=margins), \
             patch.object(features, "_compute_incumbency", return_value=incumb):
            out = build_features(gdf, state_fips, cycle)

        second = out[out["geoid"] == "4202"].iloc[0]
        assert second["margin_t0"] == 0.0
        assert second["margin_t1"] == 0.0
        assert second["margin_t2"] == 0.0
        assert second["incumbency"] == 0

    def test_polsby_popper_attached_per_row_and_in_unit_range(self):
        state_fips, cycle = "42", 2012
        cd_field = CD_FIELD[cycle]
        gdf = _state_gdf(state_fips, cd_field, ["01", "02"])

        acs = pd.DataFrame({
            "geoid": ["4201", "4202"], "total_pop": [1, 1],
            "pct_black": [0, 0], "pct_asian": [0, 0],
            "pct_hispanic": [0, 0], "pct_nonwhite": [0, 0],
            "median_income": [0.0, 0.0], "pop_deviation": [0.0, 0.0],
        })
        margins = pd.DataFrame({
            "geoid": ["4201", "4202"],
            "margin_t0": [0, 0], "margin_t1": [0, 0], "margin_t2": [0, 0],
        })
        incumb = pd.DataFrame({"geoid": ["4201", "4202"], "incumbency": [0, 0]})

        with patch.object(features, "_fetch_acs", return_value=acs), \
             patch.object(features, "_fetch_election_margins", return_value=margins), \
             patch.object(features, "_compute_incumbency", return_value=incumb):
            out = build_features(gdf, state_fips, cycle)

        pp = out["polsby_popper"]
        assert len(pp) == 2
        assert (pp > 0).all()
        assert (pp <= 1.0 + 1e-9).all()

    def test_output_crs_is_wgs84(self):
        state_fips, cycle = "42", 2012
        cd_field = CD_FIELD[cycle]
        gdf = _state_gdf(state_fips, cd_field, ["01"]).to_crs("EPSG:5070")

        acs = pd.DataFrame({
            "geoid": ["4201"], "total_pop": [1], "pct_black": [0],
            "pct_asian": [0], "pct_hispanic": [0], "pct_nonwhite": [0],
            "median_income": [0.0], "pop_deviation": [0.0],
        })
        margins = pd.DataFrame({
            "geoid": ["4201"],
            "margin_t0": [0], "margin_t1": [0], "margin_t2": [0],
        })
        incumb = pd.DataFrame({"geoid": ["4201"], "incumbency": [0]})

        with patch.object(features, "_fetch_acs", return_value=acs), \
             patch.object(features, "_fetch_election_margins", return_value=margins), \
             patch.object(features, "_compute_incumbency", return_value=incumb):
            out = build_features(gdf, state_fips, cycle)

        assert out.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Integration: real on-disk MIT Election Lab data
# ---------------------------------------------------------------------------

# Skip the whole class if the data file isn't checked out (it is gitignored).
pytestmark_integration = pytest.mark.skipif(
    not ELECTION_FILE.exists(),
    reason=f"election data file not present at {ELECTION_FILE}",
)


@pytestmark_integration
class TestRealElectionData:
    """Smoke-tests against the real ``data/raw/election/1976-2024-house.tab``.

    These guard against regressions in the file's delimiter / column casing
    and confirm that ``_fetch_election_margins`` and ``_compute_incumbency``
    can actually parse it end-to-end.
    """

    @pytest.fixture(autouse=True)
    def _chdir_repo_root(self, monkeypatch):
        # The helpers open the file via a path relative to cwd.
        monkeypatch.chdir(ELECTION_FILE.parent.parent.parent.parent)

    @pytest.mark.parametrize("cycle", sorted(CYCLE_TO_ELECTION_YEARS))
    def test_margins_runs_for_pa(self, cycle):
        out = _fetch_election_margins("42", cycle)

        assert list(out.columns) == [
            "geoid", "margin_t0", "margin_t1", "margin_t2",
        ]
        # PA has had between 17 and 19 districts across these cycles.
        assert len(out) > 0
        for col in ["margin_t0", "margin_t1", "margin_t2"]:
            non_null = out[col].dropna()
            # margins are in [-1, 1]; allow tiny float slack.
            assert ((non_null >= -1.0 - 1e-9) & (non_null <= 1.0 + 1e-9)).all()
        assert out["geoid"].str.startswith("42").all()

    @pytest.mark.parametrize("cycle", sorted(CYCLE_TO_ELECTION_YEARS))
    def test_incumbency_runs_for_pa(self, cycle):
        out = _compute_incumbency("42", cycle)
        assert list(out.columns) == ["geoid", "incumbency"]
        # incumbency is binary (0/1).
        assert set(out["incumbency"].unique()).issubset({0, 1})
        assert out["geoid"].str.startswith("42").all()

    def test_margins_runs_for_a_few_states(self):
        # Spot-check we can parse heterogeneous states without crashing.
        for fips in ["06", "36", "48"]:  # CA, NY, TX
            out = _fetch_election_margins(fips, 2020)
            assert len(out) > 0
            assert out["geoid"].str.startswith(fips).all()


# ---------------------------------------------------------------------------
# Integration: real ACS API (only when CENSUS_API_KEY is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("CENSUS_API_KEY") or os.getenv("SKIP_NETWORK_TESTS") == "1",
    reason="set CENSUS_API_KEY to run the live Census-API integration test",
)
class TestRealCensusApi:
    def test_fetch_acs_returns_real_data_for_pa(self):
        out = _fetch_acs("42", 2022)
        assert len(out) > 0
        assert out["geoid"].str.startswith("42").all()
        # Percent columns are bounded.
        for col in ["pct_black", "pct_asian", "pct_hispanic", "pct_nonwhite"]:
            non_null = out[col].dropna()
            assert ((non_null >= 0) & (non_null <= 1)).all()
