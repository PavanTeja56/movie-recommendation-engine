import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .models.user_based_cf import generate_user_based_recommendations
from .models.svd_model import generate_svd_recommendations

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def precision_at_k(rec_ids, relevant_ids, k=10):
    rec_ids = rec_ids[:k]
    if not rec_ids:
        return 0.0
    hits = sum(1 for mid in rec_ids if mid in relevant_ids)
    return hits / float(k)


def ndcg_at_k(rec_ids, relevant_ids, k=10):
    rec_ids = rec_ids[:k]
    dcg = 0.0
    for i, mid in enumerate(rec_ids):
        if mid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_single_model(generate_fn, df_test, k=10):
    """
    generate_fn(user_id, k) -> path to CSV with columns: movie_id, title, estimated_rating
    """

    rmses = []
    precisions = []
    ndcgs = []

    user_ids = df_test["user_id"].unique()
    rng = np.random.RandomState(42)
    sampled_users = rng.choice(user_ids, size=min(100, len(user_ids)), replace=False)

    for uid in sampled_users:
        user_test = df_test[df_test["user_id"] == uid]
        if user_test.empty:
            continue

        relevant = set(user_test[user_test["rating"] >= 4]["movie_id"].tolist())

        try:
            csv_path = generate_fn(target_user_id=int(uid), k=k)
        except Exception:
            continue

        rec_df = pd.read_csv(csv_path)
        if rec_df.empty:
            continue

        preds_map = dict(zip(rec_df["movie_id"], rec_df["estimated_rating"]))

        y_true, y_pred = [], []
        for _, row in user_test.iterrows():
            mid = row["movie_id"]
            if mid in preds_map:
                y_true.append(row["rating"])
                y_pred.append(preds_map[mid])

        if y_true:
            rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))

        rec_ids = rec_df["movie_id"].tolist()
        precisions.append(precision_at_k(rec_ids, relevant, k=k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k=k))

    return {
        "rmse": float(np.mean(rmses)) if rmses else 0.0,
        "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    _, df_test = train_test_split(df, test_size=0.2, random_state=42)

    user_metrics = evaluate_single_model(
        generate_user_based_recommendations, df_test, k=10
    )
    svd_metrics = evaluate_single_model(
        generate_svd_recommendations, df_test, k=10
    )

    results = {
        "user_based_cf": user_metrics,
        "svd": svd_metrics,
    }

    out_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path
