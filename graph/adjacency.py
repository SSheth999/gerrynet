import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

def adjacency(gdf: gpd.GeoDataFrame):
    """adjacency function that takes a GeoDataFrame for one state as input"""
    
    gdf = gdf.reset_index(drop=True)
    
    # creates buffered copy of gdf
    buffed_gdf = gdf.copy()
    buffed_gdf['geometry'] = buffed_gdf['geometry'].buffer(0.0001)
    
    # spatial join for overlapping pair of districts
    sjgdf = gpd.sjoin(buffed_gdf, buffed_gdf, how='inner', predicate='intersects')
    
    # remove self-joins
    sjgdf = sjgdf[sjgdf["index_left"] != sjgdf["index_right"]]