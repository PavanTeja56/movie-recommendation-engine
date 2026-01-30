# 👉 This code:

# Builds a user–movie rating matrix

# Finds users similar to the target user using cosine similarity

# Uses neighbor users’ ratings to predict ratings for unseen movies

# Recommends Top-K movies

# Saves them as a CSV

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# this file is src/models/user_based_cf.py
# go up two levels: models -> src -> project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")



def generate_user_based_recommendations(target_user_id: int = 1, k: int = 10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    # User–item rating matrix
    user_item = df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    ).fillna(0.0)

    if target_user_id not in user_item.index:
        raise ValueError(f"User {target_user_id} not found in ratings for user-based CF")

    # Cosine similarity between users
    sim_matrix = cosine_similarity(user_item)
    sim_df = pd.DataFrame(sim_matrix, index=user_item.index, columns=user_item.index)

    # Similarities to target user
    user_sims = sim_df.loc[target_user_id].drop(index=target_user_id)

    # Top similar users
    top_users = user_sims.sort_values(ascending=False).head(20)

    # Ratings of similar users
    neighbors_ratings = user_item.loc[top_users.index]

    # Weighted average prediction for each movie
    weights = top_users.values.reshape(-1, 1)
    weighted_sum = np.dot(weights.T, neighbors_ratings.values)[0]
    sim_sum = np.abs(weights).sum()
    preds = weighted_sum / np.maximum(sim_sum, 1e-8)

    pred_series = pd.Series(preds, index=user_item.columns)

    # Remove movies already rated by target user
    rated_movies = user_item.loc[target_user_id]
    unrated_mask = rated_movies == 0
    pred_series = pred_series[unrated_mask]

    # Top-k predictions
    top = pred_series.sort_values(ascending=False).head(k)

    movies = df[["movie_id", "title"]].drop_duplicates().set_index("movie_id")
    rows = []
    for mid, est in top.items():
        title = movies.loc[mid, "title"]
        rows.append(
            {
                "movie_id": int(mid),
                "title": str(title),
                "estimated_rating": float(est),
            }
        )

    out_path = os.path.join(OUTPUT_DIR, "user_based_recommendations.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
