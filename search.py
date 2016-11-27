import operator
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import heapq
import random
from intersections_and_roads import *

def cost(start_node, centroid, intersection_graph, connections, cost_function, heuristic):
    search_result = a_star_search(start_node, centroid, intersection_graph, connections, cost_function, heuristic)
    return cost_function(search_result['nodes'], search_result['connections'], intersection_graph, connections)

def simulated_annealing_b(intersection_graph, connections, cost_function, heuristic, starting_points=[]):
    if len(starting_points) < 2: 
        raise ValueError('need more than two points to find best meeting spot')
    # as a start for the simulated annealing search
    # begin by finding the centroid between all of the points
    starting_coords = [[point.get_x_y()[0],point.get_x_y()[1]] for point in starting_points]
    off_graph_centroid = np.mean(starting_coords, axis=0)

    get_closest = lambda node: np.linalg.norm(off_graph_centroid - np.array([node[1].get_x_y()[0], node[1].get_x_y()[1]]))

    centroid = min(intersection_graph.iteritems(), key=get_closest)[1]

    # Start simulated annealing around this node 
    temperature = 1e10
    gamma = 0.5
    while temperature > 1e-2:
        temperature = temperature*gamma
        # randomly select an item to swap and a swap index
        connection = random.sample(centroid.get_connections(), 1)[0]
        new_node = intersection_graph[connections[connection].get_child(centroid.id)]

        old_cost = np.sum([cost(start_node, centroid, intersection_graph, connections, cost_function, heuristic) for start_node in starting_points])
        new_cost = np.sum([cost(start_node, new_node, intersection_graph, connections, cost_function, heuristic) for start_node in starting_points])
        delta_e = old_cost - new_cost

        if delta_e >= 0:
            centroid = new_node
        elif np.random.random() < np.exp(delta_e/temperature):
            centroid = new_node

    return centroid