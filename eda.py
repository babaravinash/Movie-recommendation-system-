"""
eda.py
-------
Exploratory Data Analysis on movies.csv and ratings.csv.

Produces:
  - Console summary statistics
  - plots/eda_ratings_distribution.png
  - plots/eda_genre_distribution.png
  - plots/eda_movies_per_year.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("plots", exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
movies_df  = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── Basic stats ─────────────────────────────────────────────────────────────
print(f"\n📽  Movies   : {len(movies_df):,}")
print(f"⭐  Ratings  : {len(ratings_df):,}")
print(f"👤  Users    : {ratings_df['userId'].nunique():,}")
print(f"🎬  Avg ratings/movie : {ratings_df.groupby('movieId').size().mean():.1f}")
print(f"    Avg rating value  : {ratings_df['rating'].mean():.2f}")
print(f"    Rating std dev    : {ratings_df['rating'].std():.2f}")

# sparsity
total_possible = ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()
sparsity = 1 - len(ratings_df) / total_possible
print(f"    Matrix sparsity   : {sparsity:.2%}")

# ── 1. Ratings distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
rating_counts = ratings_df["rating"].value_counts().sort_index()
ax.bar(rating_counts.index.astype(str), rating_counts.values, color="#6C63FF", edgecolor="white", width=0.6)
ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("plots/eda_ratings_distribution.png", dpi=120)
plt.close()
print("\n📊 Saved: plots/eda_ratings_distribution.png")

# ── 2. Genre distribution ────────────────────────────────────────────────────
genre_series = movies_df["genres"].str.split("|").explode()
genre_counts = genre_series.value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color="#FF6584")
ax.set_title("Movies per Genre", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("plots/eda_genre_distribution.png", dpi=120)
plt.close()
print("📊 Saved: plots/eda_genre_distribution.png")

# ── 3. Movies per year ───────────────────────────────────────────────────────
year_counts = movies_df["year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(year_counts.index, year_counts.values, color="#43B97F", linewidth=2, marker="o", markersize=4)
ax.fill_between(year_counts.index, year_counts.values, alpha=0.2, color="#43B97F")
ax.set_title("Movies Released per Year", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plots/eda_movies_per_year.png", dpi=120)
plt.close()
print("📊 Saved: plots/eda_movies_per_year.png")

print("\n✅ EDA complete.")
