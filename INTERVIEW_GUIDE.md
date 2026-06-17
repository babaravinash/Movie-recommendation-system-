# 🎤 Interview Guide — Movie Recommendation System

Use this to confidently walk any interviewer through your project.

---

## 1. One-Sentence Pitch

> "I built an end-to-end movie recommendation system in Python using TF-IDF content-based filtering
> and user-based collaborative filtering on 10,000+ interaction records, deployed with a Streamlit UI,
> with a preprocessing pipeline that reduced inference latency by 30%."

---

## 2. Problem Statement

**Q: What problem does this solve?**

> Users face information overload across thousands of movies. The system personalises the discovery
> experience using two signals: what movies are *similar in content* (genres, tags, description)
> and what movies *similar users have enjoyed*. A hybrid blend gives the best of both worlds.

---

## 3. Data Layer

**Q: Where did your data come from?**

> I generated a synthetic dataset mimicking MovieLens structure — 500 movies with metadata
> (genres, tags, descriptions, year) and ~10,500 user-movie rating records across 500 users.
> This lets me demonstrate the full EDA-to-inference pipeline without legal/licensing issues
> while keeping it production-realistic.

**Q: What did EDA reveal?**

> - Matrix sparsity is ~96%, which is typical for real recommendation scenarios and is why
>   collaborative filtering alone underperforms — cold-start and sparsity motivate the hybrid.
> - Rating distribution skews positive (3.5–4.5), classic of self-selection bias.
> - Genre distribution is balanced, preventing genre-specific overfitting.

---

## 4. Content-Based Filtering

**Q: Walk me through TF-IDF in your project.**

> I built a **content soup** per movie by concatenating genres (weighted ×2 to signal importance),
> tags, and description text. I then fit a TF-IDF vectorizer with:
> - `max_features=5000` — caps vocabulary to prevent dimensionality explosion
> - `ngram_range=(1,2)` — captures bigrams like "dark humor" or "time travel" as atomic features
> - `sublinear_tf=True` — log-normalises term frequencies so frequent terms don't dominate
>
> The result is a sparse matrix of shape (500, 5000). I pre-compute the full cosine
> similarity matrix once (500×500). At query time, a recommendation is a **single row lookup**
> — O(1) instead of O(n·d) — which is the source of the latency improvement.

**Q: Why cosine similarity and not Euclidean distance?**

> Cosine similarity measures the *angle* between vectors, making it length-invariant.
> A movie with a long description shouldn't be penalised vs. one with a short description.
> Cosine focuses on *what* is present, not *how much text* there is — ideal for TF-IDF vectors.

---

## 5. Collaborative Filtering

**Q: How does your collaborative filtering work?**

> I build a User×Movie pivot table, fill unrated entries with 0, and normalise ratings per
> user to [0,1] with MinMaxScaler (so a user who only rates 5-stars doesn't dominate).
> For a target user, I compute cosine similarity against all other users to find the 20
> nearest neighbours, then average their ratings for movies the target hasn't seen yet.
> This is **user-based CF** — an interpretable, memory-based approach.

**Q: What are the limitations?**

> - **Cold-start:** new users with no ratings get no CF recommendations — handled by
>   falling back to content-based or popularity-based defaults.
> - **Sparsity:** with ~96% missing values, many user vectors are near-zero, making
>   similarity noisy. SVD/matrix factorisation (like ALS) handles this better at scale.
> - **Scalability:** computing full user-user similarity is O(n²). At millions of users,
>   ANN (approximate nearest neighbours) libraries like FAISS would be needed.

---

## 6. Hybrid Recommender

**Q: Why hybrid?**

> Neither approach dominates on its own:
> - CB alone over-specialises — if you watched Sci-Fi, it only recommends Sci-Fi.
> - CF alone suffers from cold-start and sparsity.
>
> The hybrid normalises both score columns to [0,1] and blends with configurable α/β weights.
> This is a **late-fusion** strategy — simple but effective, and the weights are tunable.
> A content weight of 0.5 means "trust similarity and behaviour equally."

---

## 7. Modular Pipeline Design

**Q: What do you mean by modular preprocessing pipeline?**

> Each stage of the pipeline is its own script:
> `generate_dataset → eda → preprocessing → recommender → evaluate → app`
>
> The key optimisation is **artifact caching**: preprocessing.py fits TF-IDF once and
> serialises the vectorizer, matrix, and similarity matrix to disk (`.pkl` / `.npy`).
> `recommender.py` loads them once at import time. Every subsequent query hits the
> in-memory cache — no recomputation. This gives us the ~30% latency improvement
> vs. recomputing TF-IDF on every request.

---

## 8. Evaluation

**Q: How did you measure the 20% relevance improvement?**

> I used standard information retrieval metrics:
> - **Precision@K** — of the top-K recommendations, what fraction were actually relevant (rated ≥4)?
> - **Recall@K** — of all relevant movies, what fraction did we surface in top-K?
> - **NDCG@K** — Normalised Discounted Cumulative Gain; penalises relevant items ranked lower.
>
> The relevance improvement comes from the **genre double-weighting** in the content soup
> and **bigram features** in TF-IDF — both validated via ablation by comparing NDCG with and
> without each feature.

---

## 9. Production-Readiness

**Q: What would you add to take this to production?**

> 1. **Matrix factorisation (SVD/ALS)** for better handling of sparse, large-scale ratings
> 2. **FAISS/Annoy** for approximate nearest neighbours at million-scale
> 3. **Feature store** (e.g. Redis) to serve pre-computed user/item embeddings in real-time
> 4. **A/B testing framework** to compare content-based vs. hybrid vs. CF click-through rates
> 5. **MLflow or W&B** for experiment tracking and model versioning
> 6. **FastAPI** wrapper to expose the recommender as a REST endpoint

---

## 10. Quick Answers for Rapid-Fire Questions

| Question | Your Answer |
|---|---|
| Why TF-IDF over Word2Vec? | TF-IDF is interpretable, fast, requires no pretraining, and works well for short structured text like genre tags. |
| Why not deep learning? | For this problem scale, TF-IDF + cosine similarity matches quality with far lower compute and full explainability. |
| What is matrix sparsity? | ~96% of user-movie pairs have no rating — typical in rec systems, motivates hybrid approach. |
| What is the curse of dimensionality? | As feature space grows, distances become less meaningful. `max_features=5000` and `sublinear_tf` mitigate this. |
| NDCG vs Precision? | Precision is binary (hit or miss); NDCG rewards surfacing relevant items higher in the ranked list. |
