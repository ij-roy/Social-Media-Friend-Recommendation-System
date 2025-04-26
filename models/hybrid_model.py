# FriendRecommendationSystem/models/hybrid_model.py

def combine_scores(cosine_scores, knn_scores, kmeans_scores, svd_scores, 
                   w_cosine=0.3, w_knn=0.2, w_kmeans=0.2, w_svd=0.3, top_k=5):
    """
    Combines multiple recommendation scores into a final hybrid score.
    
    Inputs:
    - cosine_scores, knn_scores, kmeans_scores, svd_scores: dictionaries {user_id: score}
    - weights for each model
    - top_k: number of top users to recommend
    
    Output:
    - List of (user_id, final_combined_score) sorted by score
    """

    final_scores = {}

    # Aggregate weighted scores
    all_user_ids = set(cosine_scores.keys()) | set(knn_scores.keys()) | set(kmeans_scores.keys()) | set(svd_scores.keys())

    for user_id in all_user_ids:
        final_scores[user_id] = (
            w_cosine * cosine_scores.get(user_id, 0) +
            w_knn * knn_scores.get(user_id, 0) +
            w_kmeans * kmeans_scores.get(user_id, 0) +
            w_svd * svd_scores.get(user_id, 0)
        )

    # Sort users by final combined score (highest first)
    sorted_final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_final_scores[:top_k]
