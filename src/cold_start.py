import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def generate_cold_start_recommendations(k: int = 10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    # Average rating per movie
    agg = (
        df.groupby("movie_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "average_rating"})
    )

    movies = df[["movie_id", "title"]].drop_duplicates()
    merged = agg.merge(movies, on="movie_id", how="left")

    merged = merged.sort_values("average_rating", ascending=False).head(k)

    out_path = os.path.join(OUTPUT_DIR, "cold_start_recommendations.csv")
    merged[["movie_id", "title", "average_rating"]].to_csv(out_path, index=False)
    return out_path
