# src/data_loader.py
import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Tuple

GENRES = ["Sci-Fi", "Action", "Drama", "Thriller", "Mystery", "Comedy"]

def generate_user_data(num_users: int = 50, catalog_path: str = "data/catalog.json") -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    genre_to_items = {g: [] for g in GENRES}
    for item in catalog:
        g = item.get("genre", "Drama")
        if g in genre_to_items:
            genre_to_items[g].append(item["id"])

    random.seed(42)
    users = []
    interactions = []

    for u_id in range(num_users):
        primary_genre = random.choice(GENRES)
        secondary_genre = random.choice([g for g in GENRES if g != primary_genre])

        # Sample 4-7 watched movies based on genre preference
        watched_candidates = genre_to_items[primary_genre] + genre_to_items[secondary_genre][:2]
        num_watched = min(len(watched_candidates), random.randint(3, 6))
        watched_ids = random.sample(watched_candidates, num_watched)

        for item_id in watched_ids:
            # 0-indexed item representation
            interactions.append((u_id, item_id - 1))

        users.append({
            "user_id": u_id,
            "username": f"user_{u_id + 1:03d}",
            "primary_genre": primary_genre,
            "preferred_genre_idx": GENRES.index(primary_genre),
            "history_item_ids": watched_ids
        })

    return users, interactions

if __name__ == "__main__":
    u, inter = generate_user_data()
    print(f"Generated {len(u)} users and {len(inter)} interactions.")
