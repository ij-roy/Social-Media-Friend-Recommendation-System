# Friend Recommendation System

A social media friend recommendation system is a project that suggests potential friends or connections to users based on their interests, demographics, and behavioral patterns. This repository contains the code and resources to build and deploy such a hybrid recommendation system using multiple machine learning techniques.

---

## Table of Contents

- Introduction  
- Features  
- Requirements  
- Dataset  
- Implementation  

---

## Introduction

The social media friend recommendation system aims to enhance user experience and engagement by suggesting connections that align with users' interests and social behavior. This project uses a hybrid approach that combines multiple models such as cosine similarity, KNN, clustering, and matrix factorization to provide personalized friend recommendations.

---

## Features

- Extraction of user features such as age, gender, and interests.  
- Calculation of user similarity using:
  - Cosine Similarity
  - K-Nearest Neighbors (KNN)
  - KMeans Clustering
  - Matrix Factorization (SVD)
- Hybrid recommendation combining multiple model outputs.  
- Graph-based and collaborative filtering-based recommendations.  
- Streamlit-based user interface to interact with the system.  
- Evaluation metrics (Precision@5, Recall@5, F1@5) to measure effectiveness.  
- Option to get recommendations for both **existing** and **new users**.

---

## Requirements

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- joblib  
- surprise  
- streamlit  
## Dataset

This project uses two publicly available social network datasets:

### 1. Facebook Ego Network Dataset

- **Source:** [Stanford SNAP - Facebook data](https://snap.stanford.edu/data/ego-Facebook.html)  
- **Description:** This dataset contains user ego networks collected from Facebook, representing friendships and circles of friends around individual users. It includes:
  - Nodes representing users (ego and their friends)
  - Edges representing friendships
  - User features such as interests and demographics can be extracted from associated files or created via feature engineering.

### 2. Pokec Social Network Dataset

- **Source:** [Stanford SNAP - Pokec data](https://snap.stanford.edu/data/soc-Pokec.html)  
- **Description:** Pokec is the most popular Slovakian social network. This dataset contains:
  - Over 1.6 million users as nodes
  - Over 30 million friendship edges
  - Rich user profile attributes including age, gender, region, and interests
  - This dataset enables extensive feature extraction and model training for friend recommendation.

### Dataset Fields (examples)

For Facebook Ego Network:

| Field            | Description                         |
|------------------|-----------------------------------|
| UserID           | Unique user identifier             |
| Friends          | List of connected user IDs        |
| Attributes       | Profile features (if available)   |

For Pokec Dataset:

| Field            | Description                      |
|------------------|--------------------------------|
| UserID           | Unique user ID                  |
| Age              | Age of the user                |
| Gender           | Gender (M/F)                  |
| Region           | Geographic region              |
| Interests        | List of interests (if available)|
| Friends          | List of user’s friends          |

Ensure the dataset files are preprocessed to extract relevant features such as user demographics, interests, and friendship connections before model training.

## Implementation

### Data Preprocessing

- Parse raw network files to extract user nodes and edges.
- Extract or engineer user features (age, gender, interests, location) from available data files.
- Generate train/test splits of friendship edges (e.g., 80% train, 20% test).
- Prepare feature matrices and similarity matrices required by the models.

### Model Training

- Train four core models using the extracted features and friendships:
  - **Cosine Similarity** on user feature vectors.
  - **KNN** trained on user feature space.
  - **KMeans Clustering** to group similar users.
  - **Matrix Factorization (SVD)** on the user-user interaction matrix.

### Hybrid Recommendation

- Combine scores from the four models with weighted averages to produce final friend recommendations.  
- The hybrid model balances content-based and collaborative filtering approaches for higher accuracy.

- The final recommendation score for a candidate friend is calculated as a weighted sum of the individual model scores:

```python
final_score = (0.3 * cosine_score) + (0.2 * knn_score) + (0.2 * kmeans_score) + (0.3 * matrix_factorization_score)



