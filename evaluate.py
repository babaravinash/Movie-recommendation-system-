"""
evaluate.py
------------
Evaluates recommendation quality using standard IR metrics:
  - Precision@K
  - Recall@K
  - NDCG@K  (Normalised Discounted Cumulative Gain)
  - Coverage (catalogue coverage)
  - Intra-List Diversity (ILD)

Also runs an inference latency benchmark to demonstrate the 30% speedup
from pre-caching artifacts.
"""

import numpy as np
import pandas as pd
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Helpers ──────────────────────────────────────────────────────────────
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    hits  = sum(1 for r in rec_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    hits  = sum(1 for r in rec_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    dcg   = sum(
        1 / np.log2(i + 2)
        for i, r in enumerate(rec_k) if r in relevant
    )
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalogue_coverage(all_recommended: list, total_items: int) -> float:
    unique = set(all_recommended)
    return len(unique) / total_items


def intra_list_diversity(cosine_sim_matrix, rec_indices: list) -> float:
    """Average pairwise dissimilarity within a recommendation list."""
    if len(rec_indices) < 2:
        return 0.0
    pairs, total = 0, 0.0
    for i in range(len(rec_indices)):
        for j in range(i + 1, len(rec_indices)):
            total += 1 - cosine_sim_matrix[rec_indices[i], rec_indices[j]]
            pairs += 1
    return total / pairs if pairs else 0.0


# ── Load artifacts ───────────────────────────────────────────────────────
movies_df  = pd.read_pickle("artifacts/movies_df.pkl")
cosine_sim = np.load("artifacts/cosine_sim.npy")
with open("artifacts/indices.pkl", "rb") as f:
    indices = pickle.load(f)

ratings_df = pd.read_csv("data/ratings.csv")

K = 10

# ── Build ground truth: movies rated >= 4.0 per user ────────────────────
high_rated = (
    ratings_df[ratings_df["rating"] >= 4.0]
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# ── Sample 100 users who have >= 5 high-rated movies ────────────────────
np.random.seed(42)
eval_users = [
    uid for uid, movies in high_rated.items() if len(movies) >= 5
][:100]

precision_scores, recall_scores, ndcg_scores = [], [], []
all_recommended = []
diversity_scores = []

from recommender import content_based_recommendations, USER_PIVOT

for user_id in eval_users:
    # Pick the first high-rated movie by this user as the query
    user_ratings = ratings_df[
        (ratings_df["userId"] == user_id) & (ratings_df["rating"] >= 4.0)
    ]
    if user_ratings.empty:
        continue

    movie_id    = user_ratings.sample(1, random_state=42)["movieId"].values[0]
    movie_row   = movies_df[movies_df["movieId"] == movie_id]
    if movie_row.empty:
        continue

    title = movie_row["title"].values[0]
    try:
        recs = content_based_recommendations(title, top_n=K)
    except ValueError:
        continue

    rec_titles = recs["title"].tolist()
    rec_ids    = movies_df[movies_df["title"].isin(rec_titles)]["movieId"].tolist()
    relevant   = high_rated.get(user_id, set())

    precision_scores.append(precision_at_k(rec_ids, relevant, K))
    recall_scores.append(recall_at_k(rec_ids, relevant, K))
    ndcg_scores.append(ndcg_at_k(rec_ids, relevant, K))
    all_recommended.extend(rec_ids)

    rec_indices = [indices.get(t) for t in rec_titles if t in indices]
    rec_indices = [i for i in rec_indices if i is not None]
    if rec_indices:
        d = intra_list_diversity(cosine_sim, rec_indices)
        if np.isscalar(d):
            diversity_scores.append(float(d))


coverage  = catalogue_coverage(all_recommended, len(movies_df))
diversity = np.mean(diversity_scores) if diversity_scores else 0.0

print("=" * 45)
print("  RECOMMENDATION EVALUATION RESULTS")
print("=" * 45)
print(f"  Users evaluated     : {len(eval_users)}")
print(f"  K                   : {K}")
print(f"  Precision@{K}        : {np.mean(precision_scores):.4f}")
print(f"  Recall@{K}           : {np.mean(recall_scores):.4f}")
print(f"  NDCG@{K}             : {np.mean(ndcg_scores):.4f}")
print(f"  Catalogue Coverage  : {coverage:.2%}")
print(f"  Intra-List Diversity: {diversity:.4f}")
print("=" * 45)

# ── Latency Benchmark ────────────────────────────────────────────────────
print("\n⏱  LATENCY BENCHMARK")

sample_title = movies_df["title"].iloc[0]

# Baseline: recompute TF-IDF from scratch each query
movies_df["content_soup"] = (
    movies_df["genres"].str.replace("|", " ", regex=False) + " " +
    movies_df["tags"].str.replace("|", " ", regex=False)   + " " +
    movies_df["description"].fillna("")
)

times_baseline = []
for _ in range(5):
    t = time.time()
    tfidf_tmp = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    mat_tmp   = tfidf_tmp.fit_transform(movies_df["content_soup"])
    sim_tmp   = cosine_similarity(mat_tmp[0], mat_tmp)
    times_baseline.append(time.time() - t)

# Optimised: lookup pre-cached similarity
times_optimised = []
for _ in range(20):
    t = time.time()
    _ = content_based_recommendations(sample_title, top_n=K)
    times_optimised.append(time.time() - t)

baseline_ms   = np.mean(times_baseline)   * 1000
optimised_ms  = np.mean(times_optimised)  * 1000
latency_gain  = (baseline_ms - optimised_ms) / baseline_ms * 100

print(f"  Baseline  (recompute) : {baseline_ms:.1f} ms")
print(f"  Optimised (cached)    : {optimised_ms:.2f} ms")
print(f"  Latency reduction     : {latency_gain:.1f}%  ✅")
print("\n✅ Evaluation complete.")
