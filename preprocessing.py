"""
preprocessing.py
-----------------
Modular preprocessing pipeline. Builds and caches:
  - TF-IDF matrix on movie content (genres + tags + description)
  - Cosine similarity matrix
  - User-Movie rating pivot table (for collaborative filtering)

Outputs saved to /artifacts/ for fast inference reuse.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

os.makedirs("artifacts", exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
movies_df  = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

# ── Step 1 : Content feature engineering ────────────────────────────────────
print("Building content features...")
movies_df["genres_clean"] = movies_df["genres"].str.replace("|", " ", regex=False)
movies_df["tags_clean"]   = movies_df["tags"].str.replace("|",   " ", regex=False)

# Weighted soup: genres repeated to boost signal
movies_df["content_soup"] = (
    movies_df["genres_clean"] + " " +
    movies_df["genres_clean"] + " " +   # double-weight genres
    movies_df["tags_clean"]   + " " +
    movies_df["description"].fillna("")
)

# ── Step 2 : TF-IDF vectorisation ───────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")
t0 = time.time()
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),     # unigrams + bigrams
    stop_words="english",
    sublinear_tf=True       # log-normalise term frequencies
)
tfidf_matrix = tfidf.fit_transform(movies_df["content_soup"])
print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}  ({time.time()-t0:.2f}s)")

# ── Step 3 : Cosine similarity ───────────────────────────────────────────────
print("Computing cosine similarity matrix...")
t0 = time.time()
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"  Similarity matrix   : {cosine_sim.shape}  ({time.time()-t0:.2f}s)")

# ── Step 4 : Build title→index lookup ───────────────────────────────────────
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

# ── Step 5 : Collaborative filtering — user-movie pivot ─────────────────────
print("Building user-movie rating matrix...")
pivot = ratings_df.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)

# Normalise ratings to [0, 1] per user (row-wise)
scaler = MinMaxScaler()
pivot_scaled = pd.DataFrame(
    scaler.fit_transform(pivot),
    index=pivot.index,
    columns=pivot.columns
)

# ── Save artifacts ───────────────────────────────────────────────────────────
print("Saving artifacts...")
with open("artifacts/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("artifacts/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

np.save("artifacts/cosine_sim.npy", cosine_sim)

with open("artifacts/indices.pkl", "wb") as f:
    pickle.dump(indices, f)

movies_df.to_pickle("artifacts/movies_df.pkl")

pivot_scaled.to_pickle("artifacts/user_movie_pivot.pkl")

print("\n✅ Preprocessing complete. Artifacts saved to /artifacts/")
print(f"   • tfidf_vectorizer.pkl")
print(f"   • tfidf_matrix.pkl")
print(f"   • cosine_sim.npy  (shape {cosine_sim.shape})")
print(f"   • indices.pkl")
print(f"   • movies_df.pkl")
print(f"   • user_movie_pivot.pkl")
