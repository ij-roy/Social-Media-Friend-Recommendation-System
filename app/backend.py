import pandas as pd
import numpy as np
import pickle
import os

# Load user profile data
PROFILES_PATH = "D:/VS Code/Semester 4/Machine Learning/FriendRecommendationSystem/data/pokec/profiles.csv"
profiles_df = pd.read_csv(PROFILES_PATH)

# Dummy function to get all other user IDs except current
def get_candidate_users(current_user_id):
    return profiles_df[profiles_df['user_id'] != current_user_id]['user_id'].values

# Dummy Cosine Similarity model
def cosine_similarity_scores(user_id, candidate_ids):
    return np.random.rand(len(candidate_ids))

# Dummy KNN model
def knn_scores(user_id, candidate_ids):
    return np.random.rand(len(candidate_ids))

# Dummy KMeans model
def kmeans_scores(user_id, candidate_ids):
    return np.random.rand(len(candidate_ids))

# Dummy Matrix Factorization model
def matrix_factorization_scores(user_id, candidate_ids):
    return np.random.rand(len(candidate_ids))

# Final recommendation function
def get_recommendations(user_id, top_n=10):
    candidate_ids = get_candidate_users(user_id)

    # Get dummy scores from each model
    cosine_scores = cosine_similarity_scores(user_id, candidate_ids)
    knn_scores_arr = knn_scores(user_id, candidate_ids)
    kmeans_scores_arr = kmeans_scores(user_id, candidate_ids)
    mf_scores = matrix_factorization_scores(user_id, candidate_ids)

    # Compute final weighted score
    final_scores = (
        0.3 * cosine_scores +
        0.2 * knn_scores_arr +
        0.2 * kmeans_scores_arr +
        0.3 * mf_scores
    )

    # Combine into a DataFrame
    results_df = pd.DataFrame({
        "user_id": candidate_ids,
        "final_score": final_scores
    })

    # Get top-N recommendations
    top_recommendations = results_df.sort_values(by="final_score", ascending=False).head(top_n)

    # Merge with profile info (optional)
    recommendations = top_recommendations.merge(profiles_df, on="user_id", how="left")

    return recommendations

# Example usage
if __name__ == "__main__":
    test_user_id = profiles_df['user_id'].iloc[0]  # Take first user as an example
    recs = get_recommendations(test_user_id)
    print("Top friend recommendations:")
    print(recs[['user_id', 'final_score'] + [col for col in profiles_df.columns if col != 'user_id']])
