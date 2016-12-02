import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import heapq

class connection:
    def __init__(self, id, source, target, distance):
        self.id = id
        self.source = source
        self.target = target
        self.distance = distance
        self.accidents = 0

    def add_accidents(self, accidents):
        self.accidents = accidents

    def get_accidents(self):
        if self.accidents:
            return self.accidents
        else:
            return None

    def get_child(self, node_id):
        if (node_id == self.source):
            return self.target
        else:
            return self.source

    def get_distance(self):
        return self.distance

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

    def id(self):
        return self.id
    #make connection_ids the road centerline IDs
    def add_connection(self, connection_id):
        if connection_id not in self.connections:
            self.connections.add(connection_id)

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
        intersection_graph[intersection.id] = node(intersection.id, intersection.P_X, intersection.P_Y)
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
                intersection_graph[node_id] = node(node_id, next_node.P_X, next_node.P_Y)

            new_node = intersection_graph[node_id]
            distance = euclidean_distance(new_node.get_x_y(), this_node.get_x_y())

            if distance < 10000: # I don't understand why.... Suggestions welcome
#             print str(street.Direction) if (str(street.Direction).strip() in ['0', '1', '-1']) else None
                this_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '1']) else None
                new_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '-1']) else None
                if street.id not in connection_dict:
                    connection_dict[street.id] = connection(street.id, this_node.id, new_node.id, distance)

def build_intersection_graph(intersections, street_centerline):
    intersection_graph = dict()
    connection_dict = dict()
    intersections.apply(follow_road, axis=1, args=[intersections, street_centerline, intersection_graph, connection_dict])
    return intersection_graph, connection_dict

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
            ax.plot(line_x, line_y)

    ax.scatter(xs, ys, s=10)

    for route in routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=5)

    for route in safe_routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, c='g', linewidth=3)

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
        weight = 1
        if connection_dict[connection_id].get_accidents() is not None:
            weight += connection_dict[connection_id].get_accidents()
        distance += connection_dict[connection_id].get_distance()*weight
    return distance

def null_heuristic(node, goal):
    return 0

def euclidean_heuristic(node, goal):
    return euclidean_distance(node.get_x_y(), goal.get_x_y())

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

                fringe.update(child, get_road_cost(route_to_goal[child.id]['nodes'], route_to_goal[child.id]['connections'], intersection_graph, connection_dict) + heuristic(child, end))
