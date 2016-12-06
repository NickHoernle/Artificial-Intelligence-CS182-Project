import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import *
import heapq
import Queue
import pdb

class connection:
    def __init__(self, id, source, target, distance):
        self.id = id
        self.source = source
        self.target = target
        self.distance = distance
        self.accidents = 0
        self.delta_elevation = 0

    def add_accidents(self, accidents):
        self.accidents = accidents

    def get_accidents(self):
        return self.accidents

    def get_child(self, node_id):
        if (node_id == self.source):
            return self.target
        else:
            return self.source

    def get_distance(self):
        return self.distance

    def set_delta_elevation(self, elevation):
        self.delta_elevation = elevation

    def get_source(self, intersection_graph):
        source = intersection_graph[self.source]
        return source

    def get_target(self, intersection_graph):
        target = intersection_graph[self.target]
        return target

class node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.connections = set()
        self.elevation = None

    def id(self):
        return self.id
    #make connection_ids the road centerline IDs
    def add_connection(self, connection_id):
        if connection_id not in self.connections:
            self.connections.add(connection_id)

    def set_elevation(self, elevation):
        self.elevation = elevation

    def get_elevation(self):
        return self.elevation

    def get_connections(self):
        return self.connections

    def get_x_y(self):
        return (self.x, self.y)

    def __str__(self):
        return '<Node> id: {}, x: {}, y: {} \nConnections: {}'.format(self.id, self.x, self.y, self.connections)


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def follow_road(intersection, intersections, street_centerline, intersection_graph, connection_dict):
    if intersection.id not in intersection_graph:
        intersection_graph[intersection.id] = node(intersection.id, intersection.geometry.x, intersection.geometry.y)
    this_node = intersection_graph[intersection.id]

    connected_streets = street_centerline[street_centerline.FromNode == intersection.NodeNumber]
    for i, street in connected_streets.iterrows():
        # create the new nodes. Add them to the node dictionary
        next_nodes = intersections[intersections.NodeNumber == street.ToNode]
        # assumption that this mapping is unique. We possibly have to verify this!
        if next_nodes.shape[0] > 0:
            next_node = next_nodes.iloc[-1] # assumption here
            node_id = next_node.id
            if node_id not in intersection_graph:
                intersection_graph[node_id] = node(node_id, next_node.geometry.x, next_node.geometry.y)

            new_node = intersection_graph[node_id]
            distance = euclidean_distance(new_node.get_x_y(), this_node.get_x_y())

            if distance < 0.01: # I don't understand why.... Suggestions welcome
                this_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '1']) else None
                new_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '-1']) else None
                if street.id not in connection_dict:
                    connection_dict[street.id] = connection(street.id, this_node.id, new_node.id, distance)

def build_intersection_graph(intersections, street_centerline, elevation, accidents):
    intersection_graph = dict()
    connection_dict = dict()
    intersections.apply(follow_road, axis=1, args=[intersections, street_centerline, intersection_graph, connection_dict])
    # remove the shitty ghost nodes
    found_nodes = dict()
    found_connections = dict()
    to_search = Queue.deque()
    to_search.append(intersection_graph['769'])
    while len(to_search) > 0:
        node = to_search.pop()

        node.set_elevation(elevation[elevation['id']==int(node.id)]['elevation'].values[0])
        found_nodes[node.id] = node
        for conn in node.get_connections():
            connection = connection_dict[conn]
            found_connections[conn] = connection
            child = intersection_graph[connection.get_child(node.id)]
            if child.id not in found_nodes:
                child.set_elevation(elevation[elevation['id']==int(child.id)]['elevation'].values[0])
                to_search.append(intersection_graph[child.id])
            connection.set_delta_elevation(child.get_elevation() - node.get_elevation())
            if conn in accidents.Street_ID.values:
                accident_num = int(accidents[accidents.Street_ID == conn].num_accidents)
                connection.add_accidents(accident_num)
    return found_nodes, found_connections

def plot_graph(intersection_graph, connection_dict, routes = [], safe_routes=[], ax = None):
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(15, 15))

    xs = [intersection_graph[key].get_x_y()[0] for key in intersection_graph]
    ys = [intersection_graph[key].get_x_y()[1] for key in intersection_graph]

    for key in intersection_graph:
        node = intersection_graph[key]
        for connection in node.get_connections():
            child = connection_dict[connection]
            line_x = [child.get_source(intersection_graph).get_x_y()[0], child.get_target(intersection_graph).get_x_y()[0]]
            line_y = [child.get_source(intersection_graph).get_x_y()[1], child.get_target(intersection_graph).get_x_y()[1]]
            ax.plot(line_x, line_y, color='#d3d3d3')

    ax.scatter(xs, ys, s=10, color='#7e7e7e')

    for route in routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=7, linestyle='dashed')

    for route in safe_routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=3)

    # plt.show()
    return ax

def plot_local_search_graph(centroid, starting_points, k_points, intersection_graph, connection_dict, routes = [], ax = None, candidate_nodes=[], fname=None):
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(15, 15))

    # clear the data to replot new data without closing the figure
        # ax.cla()
    # axis = plot_graph(intersection_graph, connection_dict,routes, [], ax=ax)
    axis = ax
    for point in candidate_nodes:
        axis.scatter(point.get_x_y()[0], point.get_x_y()[1], s=40, color='yellow')

    for point in starting_points:
        axis.scatter(point.get_x_y()[0], point.get_x_y()[1], s=60, linewidth=4, color='black', marker='x', zorder=3)

    for point in k_points:
        axis.scatter(point.get_x_y()[0], point.get_x_y()[1], s=50, color='purple', zorder=3)

    axis.scatter(centroid.get_x_y()[0], centroid.get_x_y()[1], color='red', marker='*', linewidth=5, s=90, zorder=3)

    xs = [intersection_graph[key].get_x_y()[0] for key in intersection_graph]
    ys = [intersection_graph[key].get_x_y()[1] for key in intersection_graph]

    for key in intersection_graph:
        node = intersection_graph[key]
        for connection in node.get_connections():
            child = connection_dict[connection]
            line_x = [child.get_source(intersection_graph).get_x_y()[0], child.get_target(intersection_graph).get_x_y()[0]]
            line_y = [child.get_source(intersection_graph).get_x_y()[1], child.get_target(intersection_graph).get_x_y()[1]]
            ax.plot(line_x, line_y, color='#d3d3d3', zorder=1)

    ax.scatter(xs, ys, s=10, color='#7e7e7e', zorder=1)

    for route in routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=4, linestyle='dashed', zorder=2)
    # needed to help with replotting of the data
    ax.relim()
    ax.autoscale_view(True,True,True)
    # plt.draw()
    # plt.pause(0.0001)


    if fname:
        plt.gcf()
        plt.savefig(fname=fname)
    plt.show()

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
      Thanks UC Berkeley, including a link to http://ai.berkeley.edu
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def get_road_cost(road_list, connection_list, intersection_graph, connection_dict):
    distance = 0
    for connection_id in connection_list:
        distance += connection_dict[connection_id].get_distance()
    return distance

def get_safe_road_cost(road_list, connection_list, intersection_graph, connection_dict):
    distance = 0
    for connection_id in connection_list:
        multiplier = 10000
        weight = connection_dict[connection_id].get_accidents() + 1
        distance += (max(multiplier*connection_dict[connection_id].get_distance(), 1))**weight
    return distance

def get_safe_road_cost_with_elevation(road_list, connection_list, intersection_graph, connection_dict):
    distance = 0
    for connection_id in connection_list:
        multiplier = 10000
        weight = connection_dict[connection_id].get_accidents() + 1
        distance += (max(multiplier*connection_dict[connection_id].get_distance(), 1))**weight
        distance += connection_dict[connection_id].delta_elevation
    return distance

def null_heuristic(node, goal, intersection_graph, connection_dict):
    return 0

def euclidean_heuristic(node, goal, intersection_graph, connection_dict):
    return euclidean_distance(node.get_x_y(), goal.get_x_y())

def combined_heuristic(node, goal, intersection_graph, connection_dict):
    accident_heuristic = np.min([connection_dict[c].get_accidents() for c in node.get_connections()]) + 1
    distance = euclidean_distance(node.get_x_y(), goal.get_x_y())
    elevation = goal.get_elevation() - node.get_elevation()
    return (distance*100 + elevation)**accident_heuristic

def a_star_search(start, end, intersection_graph, connection_dict, get_road_cost, heuristic=null_heuristic):
    fringe = PriorityQueue()

    discovered_nodes = set()
    route_to_goal = dict()
    route_to_goal[start.id] = {'nodes': [], 'connections': []}

    fringe.push(start, 0)

    while not fringe.isEmpty():
        node = fringe.pop()
        discovered_nodes.add(node)

        #at the goal node
        if node.id == end.id:
            return route_to_goal[node.id]

        connections = map(lambda ID: connection_dict[ID], node.get_connections())

        for connection in connections:
            child_id = connection.get_child(node.id)
            child = intersection_graph[child_id]

            #if we have not visited this node
            if not child in discovered_nodes:
                road_list = route_to_goal[node.id]['nodes'] + [child.id]
                connection_list = route_to_goal[node.id]['connections'] + [connection.id]
                cost_of_road_list = get_road_cost(road_list, connection_list, intersection_graph, connection_dict)

                # If we already have a route to this node
                if child.id in route_to_goal:
                    current_best_route = route_to_goal[child.id]
                    current_best_cost = get_road_cost(current_best_route['nodes'], current_best_route['connections'], intersection_graph, connection_dict)
#                     print 'cost', cost_of_road_list,  current_best_cost
                    if cost_of_road_list < current_best_cost:
                        route_to_goal[child.id] = {'nodes': road_list, 'connections': connection_list}
                else:
                    route_to_goal[child.id] = {'nodes': road_list, 'connections': connection_list}

                # update the fringe with this node

                fringe.update(child, get_road_cost(route_to_goal[child.id]['nodes'], route_to_goal[child.id]['connections'], intersection_graph, connection_dict) + heuristic(child, end, intersection_graph, connection_dict))
