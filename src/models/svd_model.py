import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def generate_svd_recommendations(target_user_id: int = 1, k: int = 10, n_components: int = 50):
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
        raise ValueError(f"User {target_user_id} not found in ratings for SVD model")

    # Truncated SVD factorization: user_item ≈ U * S * Vt
    n_components = min(n_components, min(user_item.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item.values)
    item_factors = svd.components_.T  # shape: n_items x n_components

    # Predicted ratings = U * V^T
    pred_matrix = np.dot(user_factors, item_factors.T)
    pred_df = pd.DataFrame(pred_matrix, index=user_item.index, columns=user_item.columns)

    user_preds = pred_df.loc[target_user_id]

    # Remove already-rated movies
    rated_mask = user_item.loc[target_user_id] > 0
    user_preds = user_preds[~rated_mask]

    top = user_preds.sort_values(ascending=False).head(k)

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

    out_path = os.path.join(OUTPUT_DIR, "svd_recommendations.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
