"""
recommender.py
---------------
Core recommendation engine.

Two strategies:
  1. Content-Based Filtering  — TF-IDF + Cosine Similarity
  2. Collaborative Filtering  — User-based cosine similarity on rating pivot
  3. Hybrid                   — Weighted blend of both scores

All artifacts are loaded once at import time → low inference latency.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# ── Load pre-built artifacts ──────────────────────────────────────────────
print("Loading recommendation artifacts...")
with open("artifacts/indices.pkl", "rb") as f:
    INDICES = pickle.load(f)

COSINE_SIM = np.load("artifacts/cosine_sim.npy")
MOVIES_DF  = pd.read_pickle("artifacts/movies_df.pkl")
USER_PIVOT = pd.read_pickle("artifacts/user_movie_pivot.pkl")
print("✅ Artifacts loaded.\n")


# ── 1. Content-Based Recommender ─────────────────────────────────────────
def content_based_recommendations(title: str, top_n: int = 10) -> pd.DataFrame:
    """
    Given a movie title, return the top-N most similar movies
    based on TF-IDF content features + cosine similarity.
    """
    if title not in INDICES:
        raise ValueError(f"Movie '{title}' not found in the database.")

    idx       = INDICES[title]
    sim_scores = list(enumerate(COSINE_SIM[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]    # exclude the query movie itself

    movie_indices = [i[0] for i in sim_scores]
    scores        = [round(i[1], 4) for i in sim_scores]

    result = MOVIES_DF.iloc[movie_indices][["title", "genres", "tags", "year"]].copy()
    result["similarity_score"] = scores
    result = result.reset_index(drop=True)
    result.index += 1
    return result


# ── 2. Collaborative Filtering Recommender ───────────────────────────────
def collaborative_recommendations(user_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Given a user_id, find the most similar users by their rating vectors,
    then recommend movies highly rated by those neighbours that the
    target user hasn't seen yet.
    """
    if user_id not in USER_PIVOT.index:
        raise ValueError(f"User {user_id} not found in the ratings database.")

    # Cosine similarity between all users
    user_sim_matrix = cosine_similarity(USER_PIVOT)
    user_sim_df     = pd.DataFrame(
        user_sim_matrix,
        index=USER_PIVOT.index,
        columns=USER_PIVOT.index
    )

    # Top-20 similar users (exclude the user themselves)
    similar_users = (
        user_sim_df[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )

    # Movies not yet rated by the target user
    target_rated = set(USER_PIVOT.loc[user_id][USER_PIVOT.loc[user_id] > 0].index)

    neighbour_ratings = USER_PIVOT.loc[similar_users]
    candidate_movies  = (
        neighbour_ratings
        .loc[:, ~neighbour_ratings.columns.isin(target_rated)]
        .mean(axis=0)
        .sort_values(ascending=False)
        .head(top_n)
    )

    movie_ids = candidate_movies.index.tolist()
    result    = MOVIES_DF[MOVIES_DF["movieId"].isin(movie_ids)][
        ["movieId", "title", "genres", "tags", "year"]
    ].copy()
    result["predicted_score"] = result["movieId"].map(
        candidate_movies.round(4)
    )
    result = result.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    result.index += 1
    return result.drop(columns=["movieId"])


# ── 3. Hybrid Recommender ────────────────────────────────────────────────
def hybrid_recommendations(
    title: str,
    user_id: int,
    top_n: int = 10,
    content_weight: float = 0.5,
    collab_weight: float  = 0.5
) -> pd.DataFrame:
    """
    Blends content-based and collaborative scores.
    Both score columns are normalised to [0, 1] before blending.
    """
    cb = content_based_recommendations(title, top_n=50)
    cf = collaborative_recommendations(user_id, top_n=50)

    # Normalise to [0, 1]
    cb["cb_norm"] = (cb["similarity_score"] - cb["similarity_score"].min()) / (
        cb["similarity_score"].max() - cb["similarity_score"].min() + 1e-9
    )
    cf["cf_norm"] = (cf["predicted_score"] - cf["predicted_score"].min()) / (
        cf["predicted_score"].max() - cf["predicted_score"].min() + 1e-9
    )

    merged = pd.merge(
        cb[["title", "genres", "tags", "year", "cb_norm"]],
        cf[["title", "cf_norm"]],
        on="title", how="outer"
    ).fillna(0)

    merged["hybrid_score"] = (
        content_weight * merged["cb_norm"] +
        collab_weight  * merged["cf_norm"]
    ).round(4)

    result = (
        merged.sort_values("hybrid_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index += 1
    return result[["title", "genres", "tags", "year", "hybrid_score"]]


# ── Quick CLI demo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_title   = MOVIES_DF["title"].iloc[0]
    sample_user_id = int(USER_PIVOT.index[0])

    print(f"📽  Content-Based for: '{sample_title}'")
    print(content_based_recommendations(sample_title, top_n=5).to_string())

    print(f"\n👤  Collaborative for User ID: {sample_user_id}")
    print(collaborative_recommendations(sample_user_id, top_n=5).to_string())

    print(f"\n🔀  Hybrid (title + user)")
    print(hybrid_recommendations(sample_title, sample_user_id, top_n=5).to_string())
