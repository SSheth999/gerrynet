import torch
import pandas as pd
from torch_geometric.data import Data
import geopandas as gpd


FEATURE_COLS =  [
    "polsby_popper",
    "total_pop",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "pct_nonwhite",
    "median_income",
    "pop_deviation",
    "margin_t0",
    "margin_t1",
    "margin_t2",
    "incumbency",
   ]

def build_graph(state_abbr: str, cycle: int) -> Data:
    """Build a graph for a given state and cycle.
    x: node features of the districts 
    edge_index: adjacency matrix of the districts
    y: label for each state & cycle if gerrymandered or not using binary classification (0 or 1)
    pos: lat/lng centroid of each district
    
    """
    graph = Data()
    # read the GeoJSON file
    gdf = gpd.read_file(f"data/processed/{state_abbr}_{cycle}.geojson")

    # x has to be a tensor for node features, set x for the graph 
    features = gdf[FEATURE_COLS].to_numpy()
    features = torch.tensor(features, dtype=torch.float)
    graph.x = features

    # edge index tensor, set edge_index for the graph
    df = pd.read_csv(f"data/processed/{state_abbr}_{cycle}_edges.csv")
    adjacency_matrix = df[["src", "dst"]].to_numpy()
    edge_index = torch.tensor(adjacency_matrix, dtype=torch.long).T.contiguous()
    graph.edge_index = edge_index

    # y labels for gerrymandering state &cycle, still a wip have to write script for creating labels.json
    # have to read labels.json and set y for the graph

    # pos has to be a tensor for coordinates, set pos for the graph
    pos = gdf.geeomtery.centroid
    coords = [[geom.x, geom.y] for geom in pos]
    coords = torch.tensor(coords, dtype=torch.float)
    graph.pos = coords

    return graph