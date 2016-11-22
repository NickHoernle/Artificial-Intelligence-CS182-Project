import numpy as np
import pandas as pd
import geopandas as gpd

class connection:
    def __init__(self, id, source, target, distance):
        self.id = id
        self.source = source
        self.target = target
        self.distance = distance
        
    def add_accidents(self, accidents):
        self.accidents = accidents
        
    def get_child(self, node_id):
        if node_id==self.source:
            return self.target
        else:
            return self.source
        

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
                this_node.add_connection(street.Street_ID)
                new_node.add_connection(street.Street_ID)
                if street.Street_ID not in connection_dict:
                    connection_dict[street.Street_ID] = connection(street.Street_ID, this_node.id, new_node.id, distance)

def build_intersection_graph(intersections, street_centerline):
    intersection_graph = dict()
    connection_dict = dict()
    intersections.apply(follow_road, axis=1, args=[intersections, street_centerline, intersection_graph, connection_dict])
    return intersection_graph