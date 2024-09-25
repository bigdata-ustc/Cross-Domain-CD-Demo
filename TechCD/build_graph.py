# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data

import networkx as nx
# import matplotlib.pyplot as plt


def build_graph(graph_type, num_nodes, data):
    edge_list = []
    if graph_type == 'prerequisite':
        with open(f'../data/{data}/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
    elif graph_type == 'similarity':
        with open(f'../data/{data}/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
                edge_list.append((int(line[1]), int(line[0])))  # Add both directions
    elif graph_type == 'exer_concept':
        with open(f'../data/{data}/Exer_Concept.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                edge_list.append((int(line[0]), int(line[1])))
                edge_list.append((int(line[1]), int(line[0])))  # Add both directions

    # Remove duplicate edges
    edge_list = list(set(edge_list))

    # Prepare edge index tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create graph data object
    data = Data(num_nodes=num_nodes, edge_index=edge_index)

    return data
