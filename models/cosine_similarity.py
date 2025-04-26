import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_friends(user_id, features_df, top_k=5):
    """
    Recommend top_k users to the given user_id based on cosine similarity.
    """
    if user_id not in features_df.index:
        raise ValueError("User ID not found in features")

    # Extract the user’s feature vector
    user_vector = features_df.loc[user_id].values.reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, features_df.values)[0]

    # Map indices to user IDs
    user_ids = features_df.index.tolist()

    # Create (user_id, similarity_score) pairs
    sim_scores = list(zip(user_ids, similarities))

    # Remove self-match
    sim_scores = [pair for pair in sim_scores if pair[0] != user_id]

    # Sort and return top_k
    top_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_k]

    return top_similar