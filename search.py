import operator
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import heapq
from intersection_graph import *

def simulated_annealing(intersection_graph, connections, cost_function, heuristic, starting_points=[]):
    if len(starting_points) < 2: 
        raise ValueError('need more than two points to find best meeting spot')

    # as a start for the simulated annealing search
    # begin by finding the centroid between all of the points
    starting_points = np.array(starting_points)
    off_graph_centroid = np.mean(starting_points)

    get_closest = lambda node: np.linalg.norm(off_graph_centroid - np.array([node[1].get_x_y()[0], node[1].get_x_y()[1]]))

    centroid = min(intersection_graph.iteritems(), key=get_closest)[1]

    # Start simulated annealing around this node 
    temperature = 1e10
    gamma = 0.999
    while temperature > 1e-2:
        temperature = temperature*gamma
        # randomly select an item to swap and a swap index
        connections = centroid.get_connections()
        connection = np.random.choice(connections)
        new_node = connections[connection].get_child(centroid.id())

        old_cost = [cost_function(a_star_search(start_node, centroid, intersection_graph, heuristic)) for start_node in starting_points]
        new_cost = [cost_function(a_star_search(start_node, new_node, intersection_graph, heuristic)) for start_node in starting_points]
        delta_e = old_cost - new_cost

        delta_e = np.sum(v[temp_bag]) - np.sum(v[annealing_bag])
        if delta_e >= 0:
            centroid = new_node
        elif np.random.random() < np.exp(delta_e/temperature):
            centroid = new_node

    return centroid
simulated_annealing(intersection_graph, connection_dict, starting_points=[[1,2],[3,4]])