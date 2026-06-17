"""
generate_dataset.py
--------------------
Generates a synthetic movie dataset (10,000+ user-movie interaction records)
mimicking a real MovieLens-style dataset.

Outputs:
  - data/movies.csv      : movie metadata (title, genres, tags, description)
  - data/ratings.csv     : user-movie ratings (userId, movieId, rating, timestamp)
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

os.makedirs("data", exist_ok=True)

# ── Movie metadata ──────────────────────────────────────────────────────────
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "Western", "Biography", "Musical"
]

TAGS = [
    "visually stunning", "plot twist", "based on true story", "cult classic",
    "slow burn", "mind-bending", "feel-good", "dark humor", "ensemble cast",
    "strong female lead", "dystopian", "time travel", "revenge", "redemption",
    "coming-of-age", "psychological", "heist", "supernatural", "space opera",
    "underdog story"
]

ADJECTIVES = [
    "The Lost", "Dark", "Rising", "Eternal", "Broken",
    "Last", "Silent", "Hidden", "Forgotten", "Final"
]
NOUNS = [
    "Hero", "City", "Empire", "Soul", "Journey",
    "Storm", "Shadow", "Legacy", "Dawn", "Realm"
]

def generate_movie_title(movie_id):
    return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {movie_id % 100 + 1}"

def generate_description(genres, tags):
    g = " and ".join(genres[:2])
    t = ", ".join(tags[:3])
    return (
        f"A gripping {g} film featuring {t}. "
        f"This movie takes the audience on an unforgettable journey "
        f"blending emotion, action, and storytelling."
    )

NUM_MOVIES = 500
movies = []
for movie_id in range(1, NUM_MOVIES + 1):
    n_genres = random.randint(1, 3)
    n_tags   = random.randint(2, 5)
    chosen_genres = random.sample(GENRES, n_genres)
    chosen_tags   = random.sample(TAGS, n_tags)
    year = random.randint(1990, 2024)
    movies.append({
        "movieId":     movie_id,
        "title":       f"{generate_movie_title(movie_id)} ({year})",
        "genres":      "|".join(chosen_genres),
        "tags":        "|".join(chosen_tags),
        "description": generate_description(chosen_genres, chosen_tags),
        "year":        year
    })

movies_df = pd.DataFrame(movies)
movies_df.to_csv("data/movies.csv", index=False)
print(f"✅ movies.csv  → {len(movies_df)} movies")

# ── User-Movie Ratings ──────────────────────────────────────────────────────
NUM_USERS   = 500
NUM_RATINGS = 10_500   # ensures 10,000+ interactions

base_time = datetime(2020, 1, 1)
ratings = []

for _ in range(NUM_RATINGS):
    user_id  = random.randint(1, NUM_USERS)
    movie_id = random.randint(1, NUM_MOVIES)
    rating   = round(random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), 1)
    ts       = base_time + timedelta(days=random.randint(0, 1460))
    ratings.append({
        "userId":    user_id,
        "movieId":   movie_id,
        "rating":    rating,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
    })

ratings_df = pd.DataFrame(ratings).drop_duplicates(subset=["userId", "movieId"])
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"✅ ratings.csv → {len(ratings_df)} interaction records")
print("\nDataset generation complete. Files saved to /data/")
