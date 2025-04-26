# FriendRecommendationSystem/models/knn_model.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

def train_knn(features_df, n_neighbors=5):
    """
    Trains a KNN model on the user feature matrix.
    """
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')  # +1 because user will be closest to themselves
    model.fit(features_df.values)
    return model

def recommend_knn(user_id, features_df, knn_model, top_k=5):
    """
    Recommends top_k users for a given user_id using trained KNN model.
    """
    if user_id not in features_df.index:
        raise ValueError("User ID not found in features")

    user_vector = features_df.loc[user_id].values.reshape(1, -1)

    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=top_k + 1)  # +1 to skip self

    # Map indices back to user IDs
    user_ids = features_df.index.tolist()
    recommended_users = []

    for idx, dist in zip(indices[0], distances[0]):
        rec_user_id = user_ids[idx]
        if rec_user_id != user_id:  # Exclude self
            recommended_users.append((rec_user_id, 1 - dist))  # 1 - distance to convert to similarity

    return recommended_users[:top_k]
