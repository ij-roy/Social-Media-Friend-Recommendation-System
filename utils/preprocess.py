# FriendRecommendationSystem/utils/preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_features(features_df):
    """
    Normalizes feature values to [0,1] range using Min-Max Scaling
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features_df)
    normalized_df = pd.DataFrame(normalized, index=features_df.index, columns=features_df.columns)
    return normalized_df
