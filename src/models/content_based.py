import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def generate_content_based_recommendations(target_user_id: int = 1, k: int = 10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    # Unique movies table
    movies = df[["movie_id", "title", "genres"]].drop_duplicates().reset_index(drop=True)

    # TF-IDF on genres
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

    # Movies rated by the target user
    user_ratings = df[df["user_id"] == target_user_id]
    if user_ratings.empty:
        raise ValueError(f"No ratings for user {target_user_id} in content-based model")

    # Take user's top-rated movies as "liked" movies
    top_user_movies = (
        user_ratings.sort_values("rating", ascending=False)
        .drop_duplicates(subset=["movie_id"])
        .head(20)
    )

    # Map their movie_ids to indices in the movies dataframe
    liked_idx = movies[movies["movie_id"].isin(top_user_movies["movie_id"])].index
    if len(liked_idx) == 0:
        raise ValueError("Could not map user liked movies to movies table")

    # Mean over liked movies -> convert to 1D array
    user_profile = tfidf_matrix[liked_idx].mean(axis=0)
    user_profile = np.asarray(user_profile).reshape(1, -1)

    sims = cosine_similarity(user_profile, tfidf_matrix).flatten()

    movies["similarity_score"] = sims

    # Remove movies already rated by the user
    seen_ids = set(user_ratings["movie_id"].tolist())
    candidates = movies[~movies["movie_id"].isin(seen_ids)]

    # Top-k by similarity_score
    top = candidates.sort_values("similarity_score", ascending=False).head(k)

    out_path = os.path.join(OUTPUT_DIR, "content_based_recommendations.csv")
    top[["movie_id", "title", "similarity_score"]].to_csv(out_path, index=False)
    return out_path
