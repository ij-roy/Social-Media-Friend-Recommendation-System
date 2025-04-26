# FriendRecommendationSystem/models/matrix_factorization_model.py

import os
import joblib
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

def build_interaction_dataframe(graph):
    """
    Converts a NetworkX friendship graph into a DataFrame with columns:
    user_id, friend_id, interaction (default = 1)
    """
    data = []
    for u, v in graph.edges():
        data.append((u, v, 1))  # Each friendship treated as implicit rating = 1
        data.append((v, u, 1))  # Since graph is undirected, add both (u,v) and (v,u)
    
    df = pd.DataFrame(data, columns=["user_id", "friend_id", "interaction"])
    return df

def train_svd(interactions_df, save_path="models/saved_models/svd_model.pkl"):
    """
    Trains a SVD model from Surprise library and saves it.
    """
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(interactions_df[["user_id", "friend_id", "interaction"]], reader)
    trainset = data.build_full_trainset()

    model = SVD(random_state=42)
    model.fit(trainset)

    # Create folder if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model
    joblib.dump(model, save_path)
    print(f"SVD model saved to {save_path}")
    
    return model
