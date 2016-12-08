import operator
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import heapq
import Queue
import matplotlib.pyplot as plt
import heapq
import random
import pdb
from intersections_and_roads import *
from shapely.geometry import *
from itertools import chain


##############################################################################
# Graph Structures to build the connected roadmap
##############################################################################

class node:
    '''
    Node class stores the intersection information, it contains
    a number of connections (roads) which connect to more terminal
    nodes.
    '''
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

class connection:
    '''
    Road segment that connects two intersections.
    '''
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

    def __str__(self):
        return '<Connection> id: {}, source: {}, target: {} \n\distance: {}, accidents: {}, delta_elev:{}'.format(self.id, self.source, self.target, self.distance, self.accidents, self.delta_elevation)

# Building the intersection graph involves folling a road segment, exploring the node at the end of that
# segment and adding it to the intersection graph. All nodes are successively explored in this manner.
class map_structure:
    def __init__(self, intersections=None, street_centerline=None, intersection_graph=None, connection_dict=None, accidents=None, elevation=None, city=None):
        if city == 'Cambridge':
            self.initialise_cambridge_input_data()
        elif city == 'San Francisco':
            self.initialise_san_fran_input_data()
        else:
            if (intersections==None or street_centerline==None or intersection_graph==None or connection_dict==None or accidents==None or elevation==None):
                print 'If city argument not given the other parameters must be set'
            else:
                self.raw_intersections = intersections
                self.raw_connections = street_centerline
                self.intersection_graph = intersection_graph
                self.road_connections = connection_dict
                self.bike_accidents = accidents
                self.elevation = elevation

    def initialise_cambridge_input_data(self,
        bike_accidents_file  = './cambridgegis_data_trans/bicycle_crashes.csv',
        elev_file  = './cambridgegis_data_trans/node_elevation_cambridge.csv',
        intersections_file  = './cambridgegis_data_trans/Intersections/TRANS_Intersections.topojson',
        street_centerline_file  = './cambridgegis_data_trans/Street_Centerlines/TRANS_Centerlines.topojson'
    ):
        bike_accidents = pd.read_csv(bike_accidents_file )
        self.bike_accidents = bike_accidents.groupby(['Street_ID'], as_index=False).size().reset_index()
        self.bike_accidents.rename(index=str, inplace=True, columns={0: 'num_accidents'})

        self.elevation = pd.read_csv(elev_file, delimiter=' ', header=None, names=['id', 'elevation'])
        self.raw_intersections = gpd.read_file(intersections_file)
        self.raw_connections = gpd.read_file(street_centerline_file)

        self.intersection_graph = dict()
        self.road_connections = dict()

        # build the intersection graph
        self.build_intersection_graph()

    def initialise_san_fran_input_data(self,
        bike_accidents_file  = "./sf_data/sf_bike_crashes.pkl",
        elev_file  = './sf_data/node_elevation_2.csv',
        intersections_file  = './sf_data/street_intersections.geojson',
        street_centerline_file  = './sf_data/San_Francisco_Basemap_Street_Centerlines.geojson'
    ):
        self.bike_accidents = pd.read_pickle(bike_accidents_file)

        self.elevation = pd.read_csv(elev_file, delimiter=' ', header=None, names=['id', 'elevation'])
        self.raw_intersections = gpd.read_file(intersections_file)
        self.raw_connections = gpd.read_file(street_centerline_file)

        self.raw_connections['ToNode'] = self.raw_connections['t_node_cnn'].astype(np.int32)
        self.raw_connections['FromNode'] = self.raw_connections['f_node_cnn'].astype(np.int32)
        self.raw_connections['id'] =  self.raw_connections['cnn'].apply(lambda x: int(float(x)))
        # Get the direction
        self.raw_connections['Direction'] = self.raw_connections['oneway'].apply(lambda x: 0 if x == 'B' else 1 if x == 'F' else -1)

        self.raw_intersections['NodeNumber'] = self.raw_intersections['cnntext'].apply(lambda x: int(float(x)))
        self.raw_intersections['id'] = self.raw_intersections['cnntext']

        self.intersection_graph = dict()
        self.road_connections = dict()

        # build the intersection graph
        self.build_intersection_graph()

    def follow_road(self, intersection):
        # if we have not encountered the intersection then add it to the graph
        if intersection.id not in self.intersection_graph:
            node_id = intersection.id
            # note we are using shapely here
            x_coord = intersection.geometry.x
            y_coord = intersection.geometry.y
            self.intersection_graph[node_id] = node(node_id, x_coord, y_coord)
            node_elevation = self.elevation[self.elevation['id']==int(node_id)]['elevation'].values[0]
            self.intersection_graph[node_id].set_elevation(node_elevation)

        this_node = self.intersection_graph[intersection.id]

        # get all of the streets that are connected to a particular intersection
        connected_streets = self.raw_connections[self.raw_connections.FromNode == intersection.NodeNumber]
        for i, street in connected_streets.iterrows():
            # find the nodes that are at the end of the streets
            next_node = None
            next_nodes = self.raw_intersections[self.raw_intersections.NodeNumber == street.ToNode]

            # assumption that this mapping is unique. We possibly have to verify this!
            if next_nodes.shape[0] >= 1:
                # get the specific node from the top of the list
                next_node = next_nodes.iloc[0]
            else:
                continue

            node_id = next_node.id

            # if this node is not in the intersection graph then add it
            if node_id not in self.intersection_graph:
                # note we are using shapely here
                x_coord = next_node.geometry.x
                y_coord = next_node.geometry.y
                new_node = node(node_id, x_coord, y_coord)
                node_elevation = self.elevation[self.elevation['id']==int(node_id)]['elevation'].values[0]
                new_node.set_elevation(node_elevation)
                self.intersection_graph[next_node.id] = new_node

            new_node = self.intersection_graph[node_id]
            distance = euclidean_distance(new_node.get_x_y(), this_node.get_x_y())

            #there is a bug in the raw data
            if distance > street.geometry.length * 10:
                continue
            #     pdb.set_trace()
            # We now make the connections and create the connection links
            this_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '1']) else None
            new_node.add_connection(street.id) if (str(street.Direction).strip() in ['0', '-1']) else None
            if street.id not in self.road_connections:
                connect = connection(street.id, this_node.id, new_node.id, distance)

                connect.set_delta_elevation(new_node.get_elevation() - this_node.get_elevation())

                if connect.id in self.bike_accidents.Street_ID.values:
                    accident_num = int(self.bike_accidents[self.bike_accidents.Street_ID == connect.id].num_accidents)
                    connect.add_accidents(accident_num)

                self.road_connections[street.id] = connect

    def build_intersection_graph(self):
        self.raw_intersections.apply(self.follow_road, axis=1)

    ########################################################
    ## Search Functions
    ########################################################
    def a_star_search(self, start, end, road_cost, heuristic=null_heuristic, return_expanded_nodes=False):
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
                if return_expanded_nodes:
                    return (route_to_goal[node.id], discovered_nodes)
                return route_to_goal[node.id]

            connections = map(lambda ID: self.road_connections[ID], node.get_connections())

            for connection in connections:
                child_id = connection.get_child(node.id)
                child = self.intersection_graph[child_id]

                #if we have not visited this node
                if not child in discovered_nodes:
                    road_list = route_to_goal[node.id]['nodes'] + [child.id]
                    connection_list = route_to_goal[node.id]['connections'] + [connection.id]
                    cost_of_road_list = road_cost(road_list, connection_list)

                    # If we already have a route to this node
                    if child.id in route_to_goal:
                        current_best_route = route_to_goal[child.id]
                        current_best_cost = road_cost(current_best_route['nodes'], current_best_route['connections'])
    #                     print 'cost', cost_of_road_list,  current_best_cost
                        if cost_of_road_list < current_best_cost:
                            route_to_goal[child.id] = {'nodes': road_list, 'connections': connection_list}
                    else:
                        route_to_goal[child.id] = {'nodes': road_list, 'connections': connection_list}

                    # update the fringe with this node
                    nodes_en_route = route_to_goal[child.id]['nodes']
                    connections_en_route = route_to_goal[child.id]['connections']
                    fringe.update(child, road_cost(nodes_en_route, connections_en_route)+heuristic(child, end))

    ########################################################
    ## Cost Functions
    ########################################################

    # wrapper cost function to calculate the cost of a
    # route via a-star, given a cost function and a heuristic
    def cost(self, start_node, centroid, cost_function, heuristic):
        search_result = self.a_star_search(start_node, centroid, cost_function, heuristic)
        nodes = search_result['nodes']
        connections = search_result['connections']
        return cost_function(nodes, connections)

    # a cost function that simply adds the total distance of that specific path
    # to the goal node
    def get_road_cost(self, road_list, connection_list):
        distance = 0
        for connection_id in connection_list:
            distance += self.road_connections[connection_id].get_distance()
        return distance

    # a cost function that includes a cost per accident that occurs on that
    # segment of road
    def get_safe_road_cost(self, road_list, connection_list):
        distance = 0
        for connection_id in connection_list:
            connection = self.road_connections[connection_id]
            # conceptually it is easier to work with numbers on the
            # order of magnitude of 1. We therefore multiply this by
            # 10000 to approximately achieve that.
            multiplier = 10000
            weight = 5*connection.get_accidents() + 1
            distance += multiplier*connection.get_distance()*weight
        return distance

    # cost function that includes a cost per accident that occurs, and
    # a cost that penalises a large change in elevation
    def get_safe_road_cost_with_elevation(self, road_list, connection_list):
        distance = 0
        for connection_id in connection_list:
            connection = self.road_connections[connection_id]
            multiplier = 10000
            weight = 5*connection.get_accidents() + 1
            distance += multiplier*connection.get_distance()*weight
            # it is important that these delta elevation and distance metrics are
            # of the same order of magnitude. It is possibly worth investigating
            # some standardisation of these terms
            distance += np.abs(connection.delta_elevation)
        return distance

    ########################################################
    ## Heuristics
    ########################################################

    # A baseline performance metric to compare the performance of the
    # other heuristics
    def null_heuristic(self, node, goal):
        return 0

    # use only euclidean distance in the heuristic. This is naive in that it
    # presumes no cost of hills and or accidents
    def euclidean_heuristic(self, node, goal):
        return euclidean_distance(node.get_x_y(), goal.get_x_y())

    # heuristic for the accidents, elevation of a node and for the euclidean distance
    def combined_heuristic(self, node, goal):
        accident_heuristic = np.min([0]+[self.road_connections[c].get_accidents() for c in node.get_connections()]) + 1
        distance = euclidean_distance(node.get_x_y(), goal.get_x_y())
        elevation = goal.get_elevation() - node.get_elevation()
        multiplier = 10000
        return (distance*multiplier + elevation)*(accident_heuristic+1)


    ########################################################
    ## Local Search Algorithms
    ########################################################

    def simulated_annealing(self, cost_function, heuristic, starting_points=[]):
        if len(starting_points) < 2:
            raise ValueError('need more than two points to find best meeting spot')
        # as a start for the simulated annealing search
        # begin by finding the centroid between all of the points
        starting_coords = [[point.get_x_y()[0],point.get_x_y()[1]] for point in starting_points]
        off_graph_centroid = np.mean(starting_coords, axis=0)

        get_closest = lambda node: np.linalg.norm(off_graph_centroid - np.array([node[1].get_x_y()[0], node[1].get_x_y()[1]]))

        centroid = min(self.intersection_graph.iteritems(), key=get_closest)[1]

        # Start simulated annealing around this node
        temperature = 1e10
        gamma = 0.5
        while temperature > 1e-2:
            temperature = temperature*gamma
            # randomly select an item to swap and a swap index
            connection = random.sample(centroid.get_connections(), 1)[0]
            new_node = self.intersection_graph[self.road_connections[connection].get_child(centroid.id)]

            old_cost = np.sum([self.cost(start_node, centroid, cost_function, heuristic) for start_node in starting_points])
            new_cost = np.sum([self.cost(start_node, new_node, cost_function, heuristic) for start_node in starting_points])
            delta_e = old_cost - new_cost

            if delta_e >= 0:
                centroid = new_node
            elif np.random.random() < np.exp(delta_e/temperature):
                centroid = new_node

        return centroid

    def k_beam_search(k, cost_function, heuristic, starting_points=[]):
        if len(starting_points) < 2:
            raise ValueError('need more than two points to find best meeting spot')

        # get coordinates of starting points
        starting_coords = np.array([[point.get_x_y()[0],point.get_x_y()[1]] for point in starting_points])

        # determine the range of x and y values
        max_x = max(starting_coords[:,0])
        min_x = min(starting_coords[:,0])
        max_y = max(starting_coords[:,1])
        min_y = min(starting_coords[:,1])

        # find all nodes within the x and y range
        node_in_target_region = lambda node: ((node[1].get_x_y()[0] > min_x) and
                                              (node[1].get_x_y()[0] < max_x) and
                                              (node[1].get_x_y()[1] < max_y) and
                                              (node[1].get_x_y()[1] > min_y))
        candidate_nodes = [node[1] for node in self.intersection_graph.iteritems() if node_in_target_region(node)]

        # select k inital random starting points
        k_points = np.random.choice(candidate_nodes, k, replace=False)
        # calculate the costs for the points
        costs = [np.sum([self.cost(start_node, k_point, cost_function, heuristic) for start_node in starting_points]) for k_point in k_points]
        # all_costs = [[cost(start_node, k_point, intersection_graph, connection_dict, cost_function, heuristic) for start_node in starting_points] for k_point in k_points]
        # costs = [(max(c) - min(c)) for c in all_costs]
        # save the min cost as the current best
        best_cost = min(costs)
        best_centroid = k_points[np.argmin(costs)]

        # helper function to get node from a connection class
        get_node = lambda (node, connection): self.intersection_graph[self.road_connections[connection].get_child(node.id)]

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
            all_successor_costs = np.array([[cost(start_node, successor_node, cost_function, heuristic) for start_node in starting_points] for successor_node in successor_nodes])
            successor_costs = np.array([np.sum(c) for c in all_successor_costs])
            # successor_costs = np.array([(max(c) - min(c)) for c in all_successor_costs])

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
                # routes, connections = get_routes_to_centroid(best_centroid, starting_points, k_points, intersection_graph, connection_dict)
                # plot_local_search_graph(best_centroid, starting_points, k_points, intersection_graph, connection_dict, routes, ax=ax, candidate_nodes=candidate_nodes)

            # if best successor is no better than current best then break and return current best
            else:
                break;
        return best_centroid, best_cost, k_points

    ##############################################################################
    # Some plotting methods
    ##############################################################################
    def plot_graph(self, routes = [], safe_routes=[], ax = None):
        if ax == None:
            fig, ax = plt.subplots(1,1, figsize=(15, 15))

        xs = [self.intersection_graph[key].get_x_y()[0] for key in self.intersection_graph]
        ys = [self.intersection_graph[key].get_x_y()[1] for key in self.intersection_graph]

        line_xs = []
        line_ys = []
        for key, node in self.intersection_graph.iteritems():
            for connection_id in node.get_connections():
                connection = self.road_connections[connection_id]
                line_x = [connection.get_source(self.intersection_graph).get_x_y()[0],
                            connection.get_target(self.intersection_graph).get_x_y()[0]]
                line_y = [connection.get_source(self.intersection_graph).get_x_y()[1],
                            connection.get_target(self.intersection_graph).get_x_y()[1]]
                line_xs.append(line_x)
                line_ys.append(line_y)

                ax.plot(line_x, line_y, color='#d3d3d3')
        ax.scatter(xs, ys, s=10, color='#7e7e7e')

        for route in routes:
            xs = [self.intersection_graph[node].get_x_y()[0] for node in route]
            ys = [self.intersection_graph[node].get_x_y()[1] for node in route]
            ax.plot(xs, ys, linewidth=7, linestyle='dashed')

        for route in safe_routes:
            xs = [self.intersection_graph[node].get_x_y()[0] for node in route]
            ys = [self.intersection_graph[node].get_x_y()[1] for node in route]
            ax.plot(xs, ys, linewidth=3)

        return ax

    def plot_local_search_graph(centroid, starting_points, k_points, routes = [], ax = None, candidate_nodes=[], fname=None):
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

        xs = [self.intersection_graph[key].get_x_y()[0] for key in self.intersection_graph]
        ys = [self.intersection_graph[key].get_x_y()[1] for key in self.intersection_graph]

        for key, node in self.intersection_graph.iteritems():
            for connection in node.get_connections():
                child = self.road_connections[connection]
                line_x = [child.get_source(self.intersection_graph).get_x_y()[0], child.get_target(self.intersection_graph).get_x_y()[0]]
                line_y = [child.get_source(self.intersection_graph).get_x_y()[1], child.get_target(self.intersection_graph).get_x_y()[1]]
                ax.plot(line_x, line_y, color='#d3d3d3', zorder=1)

        ax.scatter(xs, ys, s=10, color='#7e7e7e', zorder=1)

        for route in routes:
            xs = [self.intersection_graph[node].get_x_y()[0] for node in route]
            ys = [self.intersection_graph[node].get_x_y()[1] for node in route]
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

##############################################################################
# Useful methods and classes for the search algorithms
##############################################################################

# We require the euclidean distance between two points. This is a
# helper for that calculation
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

#A* search requires a Priority queue implementation that is able to conditionally
# update an entry IFF the new value presents a lower cost.
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

# helper function to calculate the routes of all the starting points to the centroid
def get_routes_to_centroid(best_centroid, starting_points, intersection_graph, connection_dict):
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


if __name__ == '__main__':
    map_structure(city='Cambridge')
