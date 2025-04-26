# FriendRecommendationSystem/utils/metrics.py

def precision_at_k(recommended_users, true_friends, k):
    recommended_top_k = recommended_users[:k]
    relevant = sum([1 for user in recommended_top_k if user in true_friends])
    return relevant / k

def recall_at_k(recommended_users, true_friends, k):
    recommended_top_k = recommended_users[:k]
    relevant = sum([1 for user in recommended_top_k if user in true_friends])
    return relevant / len(true_friends) if true_friends else 0

def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def hit_rate_at_k(recommended_users, true_friends, k):
    recommended_top_k = recommended_users[:k]
    return int(any(user in true_friends for user in recommended_top_k))

def mean_reciprocal_rank(recommended_users, true_friends):
    for idx, user in enumerate(recommended_users, start=1):
        if user in true_friends:
            return 1 / idx
    return 0
