# FriendRecommendationSystem/utils/data_loader.py

import pandas as pd
import networkx as nx

def load_features(features_path):
    """
    Loads user features from features.txt
    """
    features = pd.read_csv(features_path, sep=' ', header=None)
    features = features.rename(columns={0: 'user_id'})  # First column = user_id
    features.set_index('user_id', inplace=True)
    return features

def load_edges(edges_path):
    """
    Loads friendship edges into a NetworkX graph
    """
    G = nx.Graph()
    with open(edges_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G