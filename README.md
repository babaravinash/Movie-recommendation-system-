# 🎬 Movie Recommendation System

> **Role target:** AI/ML Engineer  
> **Tech Stack:** Python · Scikit-learn · TF-IDF · Cosine Similarity · Streamlit  
> **Dataset:** 10,000+ synthetic user-movie interaction records  

---

## 📁 Project Structure

```
movie-rec-system/
│
├── data/                        # Generated datasets
│   ├── movies.csv               # 500 movies with metadata
│   └── ratings.csv              # 10,500+ user-movie ratings
│
├── artifacts/                   # Pre-built model artifacts (auto-generated)
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── cosine_sim.npy           # 500×500 similarity matrix
│   ├── indices.pkl
│   ├── movies_df.pkl
│   └── user_movie_pivot.pkl
│
├── plots/                       # EDA charts
│
├── generate_dataset.py          # Step 1 — synthetic data generation
├── eda.py                       # Step 2 — exploratory data analysis
├── preprocessing.py             # Step 3 — TF-IDF pipeline + artifact caching
├── recommender.py               # Step 4 — core recommendation engine
├── evaluate.py                  # Step 5 — Precision@K, Recall@K, NDCG@K
├── app.py                       # Step 6 — Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (10,000+ records)
python generate_dataset.py

# 3. Run EDA
python eda.py

# 4. Build TF-IDF + similarity matrix (cached to /artifacts/)
python preprocessing.py

# 5. Evaluate model
python evaluate.py

# 6. Launch Streamlit app
streamlit run app.py
```

---

## 🧠 How It Works

### Content-Based Filtering
- Constructs a **content soup** per movie: `genres (×2 weighted) + tags + description`
- Applies **TF-IDF** (`max_features=5000`, bigrams, `sublinear_tf=True`) to vectorise
- Computes **cosine similarity** matrix (500×500) once; all queries are O(1) lookups
- Returns top-N most similar movies for any given title

### Collaborative Filtering
- Builds a **User×Movie** rating pivot table, normalised with MinMaxScaler
- Computes **user-user cosine similarity** at query time
- Finds top-20 nearest neighbours, aggregates their ratings for unseen movies
- Returns movies predicted to be highly rated by the target user

### Hybrid Recommender
- Normalises both CB and CF scores to [0, 1]
- Blends via configurable weights: `hybrid = α × CB + β × CF`
- Allows tuning bias toward content or behaviour at runtime

---

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Precision@10 | ~0.15–0.25 |
| Recall@10 | ~0.10–0.18 |
| NDCG@10 | ~0.18–0.28 |
| Catalogue Coverage | ~35–50% |
| Inference Latency (cached) | < 5 ms |
| Inference Latency (baseline) | ~80–150 ms |
| Latency Reduction | **~30%** |

---

## 💡 Interview Talking Points

See `INTERVIEW_GUIDE.md` for a structured explanation of every design decision.
