import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .data_preprocessing import build_processed_movies
from .cold_start import generate_cold_start_recommendations
from .models.svd_model import generate_svd_recommendations

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

app = FastAPI(title="Movie Recommendation API")


def load_processed_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "processed_movies.csv")
    if not os.path.exists(path):
        build_processed_movies()
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok"}

def get_svd_recommendations_for_user(user_id: int, k: int = 10) -> pd.DataFrame:
    df = load_processed_data()
    if user_id not in df["user_id"].unique():
        raise KeyError("user_not_found")

    csv_path = generate_svd_recommendations(target_user_id=user_id, k=k)
    recs = pd.read_csv(csv_path)
    # Ensure correct columns
    return recs[["movie_id", "title", "estimated_rating"]]


def get_cold_start_top_k(k: int = 10) -> pd.DataFrame:
    # This will generate the CSV if needed
    csv_path = generate_cold_start_recommendations(k=k)
    cold = pd.read_csv(csv_path)
    # Must be sorted by average_rating desc per spec
    cold = cold.sort_values("average_rating", ascending=False).head(k)
    return cold[["movie_id", "title", "average_rating"]]

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    k = 10

    try:
        recs = get_svd_recommendations_for_user(user_id=user_id, k=k)
        items = [
            {
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "estimated_rating": float(row["estimated_rating"]),
            }
            for _, row in recs.iterrows()
        ]
        return {"user_id": user_id, "recommendations": items}
    except KeyError:
        # Cold-start: user not in training data
        cold = get_cold_start_top_k(k=k)
        items = [
            {
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "estimated_rating": float(row["average_rating"]),
            }
            for _, row in cold.iterrows()
        ]
        return {"user_id": user_id, "recommendations": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
