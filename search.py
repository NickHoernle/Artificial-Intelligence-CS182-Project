import operator
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import heapq
import random
import pdb
from intersections_and_roads import *
from shapely.geometry import *
from itertools import chain

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

def k_beam_search(k, intersection_graph, connection_dict, cost_function, heuristic, starting_points=[]):
    if len(starting_points) < 2:
        raise ValueError('need more than two points to find best meeting spot')

    # initialise plot
    plt.ion()
    fig, ax = plt.subplots(1,1, figsize=(15, 15))
    ax.set_autoscale_on(True)

    # get coordinates of starting points
    starting_coords = np.array([[point.get_x_y()[0],point.get_x_y()[1]] for point in starting_points])

    # determine the range of x and y values
    max_x = max(starting_coords[:,0])
    min_x = min(starting_coords[:,0])
    max_y = max(starting_coords[:,1])
    min_y = min(starting_coords[:,1])

    # find all nodes within the x and y range
    node_in_target_region = lambda node: ((node[1].get_x_y()[0] > min_x) and (node[1].get_x_y()[0] < max_x) and (node[1].get_x_y()[1] < max_y) and (node[1].get_x_y()[1] > min_y))
    candidate_nodes = [node[1] for node in intersection_graph.iteritems() if node_in_target_region(node)]

    # select k inital random starting points
    k_points = np.random.choice(candidate_nodes, k, replace=False)
    # calculate the costs for the points
    # costs = [np.sum([cost(start_node, k_point, intersection_graph, connection_dict, cost_function, heuristic) for start_node in starting_points]) for k_point in k_points]
    all_costs = [[cost(start_node, k_point, intersection_graph, connection_dict, cost_function, heuristic) for start_node in starting_points] for k_point in k_points]
    costs = [(max(c) - min(c)) for c in all_costs]
    # save the min cost as the current best
    best_cost = min(costs)
    best_centroid = k_points[np.argmin(costs)]

    # helper function to get node from a connection class
    get_node = lambda (node, connection): intersection_graph[connection_dict[connection].get_child(node.id)]

    # counter for keepting track of iterations
    i = 0

    # continue iterating until a successor cost is not less than the current best
    while True:
        print 'iteration ',  i, 'best cost', best_cost
        i+=1

        # generate all successor connections and flatten into single list
        successor_connections = list(chain.from_iterable([[(k_node, connection) for connection in k_node.get_connections()] for k_node in k_points]))

        # generate all successor nodes
        successor_nodes = np.array([get_node((node, connection)) for (node, connection) in successor_connections])

        # evaluate costs of all successors
        all_successor_costs = np.array([[cost(start_node, successor_node, intersection_graph, connection_dict, cost_function, heuristic) for start_node in starting_points] for successor_node in successor_nodes])
        successor_costs = np.array([(max(c) - min(c)) for c in all_successor_costs])

        # retain best k successors
        best_k_indices = np.argsort(successor_costs, axis=0)[:k]
        best_k_costs = successor_costs[best_k_indices]

        # update k centroids
        k_points = successor_nodes[best_k_indices]

        # update current best centroid and best cost
        if best_k_costs[0] < best_cost:
            best_cost = best_k_costs[0]
            best_centroid = k_points[0]

            # replot the routes and centroid locations
            routes, connections = get_routes_to_centroid(best_centroid, starting_points, k_points, intersection_graph, connection_dict)
            plot_local_search_graph(best_centroid, starting_points, k_points, intersection_graph, connection_dict, routes, ax=ax, candidate_nodes=candidate_nodes)

        # if best successor is no better than current best then break and return current best
        else:
            break;
    return best_centroid, best_cost, k_points

# helper function to calculate the routes of all the starting points to the centroid
def get_routes_to_centroid(best_centroid, starting_points, k_points, intersection_graph, connection_dict):
    routes = []
    connections = []
    for start in starting_points:
        route = a_star_search(start, best_centroid, intersection_graph, connection_dict, get_road_cost)
        if route:
            routes.append(route['nodes'])
            connections.append(route['connections'])
        else:
            # if there is no route then append empty list
            routes.append([])
            connections.append([])
    return routes, connections
