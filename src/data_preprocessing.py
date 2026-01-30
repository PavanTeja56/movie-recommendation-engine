import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_movies.csv")


def load_raw_movielens():
    u_data_path = os.path.join(DATA_DIR, "u.data")
    u_item_path = os.path.join(DATA_DIR, "u.item")

    # u.data: user_id \t movie_id \t rating \t timestamp
    ratings = pd.read_csv(
        u_data_path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )

    # u.item: movie_id | title | release_date | video_release | imdb_url | genres(19 flags)
    items = pd.read_csv(
        u_item_path,
        sep="|",
        header=None,
        encoding="latin-1",
        engine="python",
    )

    # First 2 columns are movie_id, title; last 19 are genre flags
    items = items.rename(columns={0: "movie_id", 1: "title"})
    genre_cols = items.columns[-19:]

    # Build pipe-separated genres string
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]

    items["genres"] = (
        items[genre_cols]
        .apply(
            lambda row: "|".join(
                [g for g, flag in zip(genre_names, row.values) if flag == 1]
            ),
            axis=1,
        )
    )

    items_simple = items[["movie_id", "title", "genres"]]
    return ratings, items_simple


def build_processed_movies():
    ratings, items = load_raw_movielens()

    df = ratings.merge(items, on="movie_id", how="left")

    df = df[["user_id", "movie_id", "rating", "title", "genres"]]
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(int)

    df.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH