
from shapely.geometry import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from intersections_and_roads import *
from search import *


# Read in the geolocation data
intersections = gpd.read_file('./cambridgegis_data_trans/Intersections/TRANS_Intersections.topojson')
street_centerline = gpd.read_file('./cambridgegis_data_trans/Street_Centerlines/TRANS_Centerlines.topojson')
intersection_graph, connection_dict = build_intersection_graph(intersections=intersections, street_centerline=street_centerline)

p1 = intersection_graph[np.random.choice(intersection_graph.keys())]
p2 = intersection_graph[np.random.choice(intersection_graph.keys())]
p3 = intersection_graph[np.random.choice(intersection_graph.keys())]
p4 = intersection_graph[np.random.choice(intersection_graph.keys())]
p5 = intersection_graph[np.random.choice(intersection_graph.keys())]

starting_points = [p1,p2,p3, p4, p5]
best_centroid, best_cost, k_points = local_beam_search(10, intersection_graph, connection_dict, get_road_cost, euclidean_heuristic, starting_points=starting_points)
