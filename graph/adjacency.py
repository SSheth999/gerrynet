import geopandas as gpd
import pandas as pd

# Equal-area projection for the contiguous U.S.; gives meter-accurate
# buffers and lengths. For AK/HI/PR you'd ideally swap in a local CRS,
# but EPSG:5070 is fine within a single state for relative comparison.
PROJECTED_CRS = "EPSG:5070"

# Buffer in meters, used only to absorb tiny gaps between district polygons
# so that truly adjacent districts get matched by `intersects`.
BUFFER_METERS = 10.0


def adjacency(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Adjacency edges (undirected, mirrored) for one state's districts.

    Returns a DataFrame with columns ``src``, ``dst``, ``border_length`` where
    ``border_length`` is in meters under :data:`PROJECTED_CRS`. Each undirected
    adjacency is represented as two rows (i->j and j->i).
    """
    gdf = gdf.reset_index(drop=True).to_crs(PROJECTED_CRS)

    buffed_gdf = gdf.copy()
    buffed_gdf["geometry"] = buffed_gdf.geometry.buffer(BUFFER_METERS)

    sjgdf = gpd.sjoin(buffed_gdf, buffed_gdf, how="inner", predicate="intersects")
    sjgdf = sjgdf.reset_index().rename(columns={"index": "index_left"})
    sjgdf = sjgdf[sjgdf["index_left"] < sjgdf["index_right"]]

    if sjgdf.empty:
        return pd.DataFrame(columns=["src", "dst", "border_length"])

    left_geom = gdf.geometry.iloc[sjgdf["index_left"].to_numpy()].reset_index(drop=True)
    right_geom = gdf.geometry.iloc[sjgdf["index_right"].to_numpy()].reset_index(drop=True)
    border_length = left_geom.intersection(right_geom).length.to_numpy()

    fwd = pd.DataFrame({
        "src": sjgdf["index_left"].to_numpy(dtype=int),
        "dst": sjgdf["index_right"].to_numpy(dtype=int),
        "border_length": border_length,
    })
    fwd = fwd[fwd["border_length"] > 0]

    rev = fwd.rename(columns={"src": "dst", "dst": "src"})[["src", "dst", "border_length"]]
    return pd.concat([fwd, rev], ignore_index=True)
