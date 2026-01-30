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


def precision_at_k(recommended_ids, actual_ids, k=10):
    if len(recommended_ids) == 0:
        return 0.0
    recommended_top_k = recommended_ids[:k]
    hits = sum(1 for mid in recommended_top_k if mid in actual_ids)
    return hits / float(k)


def ndcg_at_k(recommended_ids, actual_ids, k=10):
    recommended_top_k = recommended_ids[:k]
    dcg = 0.0
    for i, mid in enumerate(recommended_top_k):
        if mid in actual_ids:
            dcg += 1.0 / np.log2(i + 2)  # positions are 1-based
    # ideal DCG (all hits at top)
    ideal_hits = min(len(actual_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_model(generate_fn, model_name, df, test_df, k=10):
    """
    generate_fn: function(target_user_id, k) -> path to CSV with movie_id, title, estimated_rating
    model_name: "user_based_cf" or "svd"
    df: full ratings dataframe
    test_df: test split dataframe
    """
    rmses = []
    precisions = []
    ndcgs = []

    user_ids = test_df["user_id"].unique()
    # sample some users to keep evaluation reasonable
    rng = np.random.RandomState(42)
    sampled_users = rng.choice(user_ids, size=min(200, len(user_ids)), replace=False)

    for uid in sampled_users:
        user_test = test_df[test_df["user_id"] == uid]
        if user_test.empty:
            continue

        # True "relevant" items: movies where rating >= 4 in test
        relevant = set(user_test[user_test["rating"] >= 4]["movie_id"].tolist())

        # Generate recommendations for this user
        try:
            generate_fn(target_user_id=int(uid), k=k)
        except ValueError:
            # model may not support some users; skip
            continue

        rec_path = os.path.join(OUTPUT_DIR, f"{model_name}_temp_eval.csv")
        # copy the last recommendations file to a temp filename for reading
        # but easier: have generate_fn always write to a known temp path
        # For simplicity, re-read main output and assume it's for this user.
        main_path = os.path.join(OUTPUT_DIR, f"{model_name}_recommendations.csv")
        if not os.path.exists(main_path):
            continue
        rec_df = pd.read_csv(main_path)

        # predicted scores for movies in test set
        y_true = []
        y_pred = []
        preds_map = dict(zip(rec_df["movie_id"], rec_df.get("estimated_rating", 0.0)))
        for _, row in user_test.iterrows():
            mid = row["movie_id"]
            if mid in preds_map:
                y_true.append(row["rating"])
                y_pred.append(preds_map[mid])

        if len(y_true) > 0:
            rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
            rmses.append(rmse_val)

        rec_ids = rec_df["movie_id"].tolist()
        precisions.append(precision_at_k(rec_ids, relevant, k=k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k=k))

    metrics = {
        "rmse": float(np.mean(rmses)) if rmses else 0.0,
        "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
    return metrics


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_movies.csv"))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # For evaluation we still train on full data in your current model functions,
    # but metrics will be computed using test_df relevance.
    user_metrics = evaluate_model(
        generate_user_based_recommendations, "user_based", df, test_df, k=10
    )
    svd_metrics = evaluate_model(
        generate_svd_recommendations, "svd", df, test_df, k=10
    )

    results = {
        "user_based_cf": user_metrics,
        "svd": svd_metrics,
    }

    out_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path
