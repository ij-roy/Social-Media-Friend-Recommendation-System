# FriendRecommendationSystem/models/kmeans_model.py

import os
import joblib
from sklearn.cluster import KMeans

def train_kmeans(features_df, n_clusters=10, save_path="models/saved_models/kmeans_model.pkl"):
    """
    Trains a KMeans clustering model and saves it to disk.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(features_df.values)

    # Create saved_models folder if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model
    joblib.dump(model, save_path)
    print(f"KMeans model saved to {save_path}")
    
    return model

def predict_cluster(user_id, features_df, kmeans_model):
    """
    Predicts the cluster of a given user.
    """
    if user_id not in features_df.index:
        raise ValueError("User ID not found in features")

    user_vector = features_df.loc[user_id].values.reshape(1, -1)
    cluster_label = kmeans_model.predict(user_vector)[0]

    return cluster_label
