import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def generate_item_based_recommendations(target_user_id: int = 1, k: int = 10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    # User–item matrix
    user_item = df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    ).fillna(0.0)

    if target_user_id not in user_item.index:
        raise ValueError(f"User {target_user_id} not found in ratings for item-based CF")

    # Item–item cosine similarity
    item_matrix = user_item.T  # shape: n_items x n_users
    sim_matrix = cosine_similarity(item_matrix)
    sim_df = pd.DataFrame(
        sim_matrix, index=item_matrix.index, columns=item_matrix.index
    )

    target_ratings = user_item.loc[target_user_id]

    scores = {}
    for movie_id, rating in target_ratings[target_ratings > 0].items():
        # similarity vector for this movie to all others
        sims = sim_df.loc[movie_id]
        for other_id, sim_val in sims.items():
            if other_id == movie_id:
                continue
            if target_ratings[other_id] > 0:
                continue  # already rated
            scores.setdefault(other_id, []).append(sim_val * rating)

    if not scores:
        raise ValueError("No candidate items for item-based CF")

    # Aggregate scores
    agg_scores = {
        mid: float(np.sum(vals)) for mid, vals in scores.items()
    }

    # Top-k
    top_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    movies = df[["movie_id", "title"]].drop_duplicates().set_index("movie_id")
    rows = []
    for mid, score in top_items:
        title = movies.loc[mid, "title"]
        rows.append(
            {
                "movie_id": int(mid),
                "title": str(title),
                "estimated_rating": float(score),
            }
        )

    out_path = os.path.join(OUTPUT_DIR, "item_based_recommendations.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
