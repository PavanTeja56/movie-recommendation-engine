# Movie Recommendation Engine (MovieLens 100k)

A production‑style movie recommendation engine built on the MovieLens 100k dataset.
It implements multiple recommendation strategies (user‑based CF, item‑based CF, matrix factorization, and content‑based) and exposes the best‑performing model via a Dockerized FastAPI service.
​

## 1. Project Overview
This project demonstrates how modern platforms (e.g., streaming or e‑commerce sites) can recommend relevant content from a large catalog.
​
It covers the full lifecycle of a recommendation system:

- Data preprocessing for MovieLens 100k

- Multiple recommendation algorithms

- Offline evaluation with rating and ranking metrics

- Cold‑start handling for new users

- A production‑oriented API layer with Docker and Docker Compose for one‑command startup.
​

## 2. Features
Models
All models train on the preprocessed ratings and generate top‑10 recommendations for a given user (user_id = 1 for the contract files).
​

### User‑Based Collaborative Filtering

- k‑Nearest Neighbors over user rating profiles

- Output: output/user_based_recommendations.csv

- Columns: movie_id, title, estimated_rating (10 rows).
​

### Item‑Based Collaborative Filtering

- Similarity between items based on user ratings

- Output: output/item_based_recommendations.csv

- Columns: movie_id, title, estimated_rating.
​

### Matrix Factorization (SVD‑style)

- Low‑rank approximation of the user–item matrix using truncated SVD

- Output: output/svd_recommendations.csv

- Columns: movie_id, title, estimated_rating.
​

#### Content‑Based Filtering

- TF‑IDF on movie genres and cosine similarity

- Output: output/content_based_recommendations.csv

- Columns: movie_id, title, similarity_score.
​

**Evaluation**
Offline evaluation for User‑Based CF and SVD:
​

- Train/test split on ratings

- RMSE for rating prediction accuracy

- Precision@10 and NDCG@10 for top‑N ranking quality

- Output: output/evaluation_metrics.json with structure:

json
"""
{
  "user_based_cf": {
    "rmse": 0.0,
    "precision_at_10": 0.0,
    "ndcg_at_10": 0.0
  },
  "svd": {
    "rmse": 0.0,
    "precision_at_10": 0.0,
    "ndcg_at_10": 0.0
  }
}
"""

**Cold‑Start Handling**
For users with no history, the system falls back to most popular movies by average rating across all users.
​

- Output file: output/cold_start_recommendations.csv

- Columns: movie_id, title, average_rating (10 rows, sorted by average_rating descending).
​

## 3. Tech Stack
- Language: Python 3.10

- Core libraries:

    - Data: pandas, numpy

    - Models: scikit-learn (k‑NN, TruncatedSVD, TF‑IDF, cosine similarity)

- API: FastAPI + Uvicorn

- Containerization: Docker, Docker Compose

- Dataset: MovieLens 100k (u.data, u.item, u.user).
​

## 4. Project Structure
"""
/
├── data/                     # Raw MovieLens files and processed_movies.csv
├── output/                   # Generated outputs and metrics
├── src/                      # All Python source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── main.py               # Orchestration: preprocessing, models, eval, cold-start
│   ├── cold_start.py
│   ├── evaluation.py
│   ├── api.py                # FastAPI app (health + recommendations)
│   └── models/
│       ├── __init__.py
│       ├── user_based_cf.py
│       ├── item_based_cf.py
│       ├── svd_model.py
│       └── content_based.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
"""

## 5. Getting Started
### 5.1 Prerequisites
- Docker and Docker Compose installed

- (Optional for local, non‑Docker run) Python 3.10 with pip.
​

### 5.2 Environment Variables
Copy the example env file and adjust if needed:

""" 
cp .env.example .env
""" 

Default values:

""" 
API_PORT=8000
DEFAULT_USER_ID=1
"""

- API_PORT: external port exposed by the API container.

- DEFAULT_USER_ID: default user for certain operations (not required to run the API).
​

## 6. Running the Project
### 6.1 One‑Command Docker Setup (recommended)
From the project root:

"""
docker-compose up --build
"""

What this does:

1.Builds the Docker image from Dockerfile.

2.Runs python -m src.main inside the container, which:

    - Processes MovieLens data into data/processed_movies.csv

    - Generates recommendation CSVs for all four models

    - Computes evaluation metrics

    - Generates cold‑start recommendations.
​

3.Starts the FastAPI server with Uvicorn on port 8000 inside the container.

4.Healthcheck periodically calls GET /health to mark the container as healthy.
​

To stop:

"""
docker-compose down
"""

### 6.2 Local (non‑Docker) run (optional)
Inside an activated Python environment:

"""
pip install -r requirements.txt
python -m src.main
"""

This will generate all outputs in the output/ directory but will not start the API automatically.
​
To run the API locally:

"""
uvicorn src.api:app --reload
"""

### 7. API Usage
Once the container (or local Uvicorn) is running, the API is available at:

"""
http://localhost:API_PORT
"""

By default API_PORT=8000.

### 7.1 Health Check
"""
GET /health
"""

Response

"""
{
  "status": "ok"
}
"""

This endpoint is also used by the container healthcheck.
​

### 7.2 Get Recommendations for a User
"""
GET /recommendations/{user_id}
"""

Path parameter:

- user_id (integer): ID of the user for whom to generate recommendations.
​

Success (known user)

- Uses the SVD model trained on MovieLens data.
​

Example request:

"""
GET /recommendations/1
"""

Example response (truncated):

"""
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 318,
      "title": "Schindler's List (1993)",
      "estimated_rating": 3.37
    },
    ...
  ]
}
"""
There are always exactly 10 recommendation objects.
​

Success (unknown user / cold‑start)

If user_id is not present in the training data, the endpoint returns the top‑10 most popular movies by average rating, still under the estimated_rating field.
​

Example:
"""
GET /recommendations/9999
"""
returns:

"""
{
  "user_id": 9999,
  "recommendations": [
    {
      "movie_id": 1189,
      "title": "Prefontaine (1997)",
      "estimated_rating": 5.0
    },
    ...
  ]
}
"""

### 8. Generated Outputs
After running python -m src.main (locally or via Docker), the output/ directory contains:

- user_based_recommendations.csv – User‑based CF, 10 rows.
​

- item_based_recommendations.csv – Item‑based CF, 10 rows.
​

- svd_recommendations.csv – SVD model, 10 rows.
​

- content_based_recommendations.csv – Content‑based, 10 rows.
​

- evaluation_metrics.json – RMSE, Precision@10, NDCG@10 for user‑based CF and SVD.
​

- cold_start_recommendations.csv – Popularity‑based cold‑start list, 10 rows.
​

These files follow the column and schema contracts defined in the original task document.
​

### 9. Design Notes and Trade‑offs
- The project uses Scikit‑learn implementations (k‑NN, TruncatedSVD, TF‑IDF) instead of the Surprise library, which still satisfies the requirement to implement the algorithms while avoiding native build issues on some platforms.
​

- Models are trained and outputs generated at container startup so the API serves recommendations from precomputed results without retraining on each request.
​

- The cold‑start solution is intentionally simple (average rating popularity), mirroring a common baseline strategy in real systems.
