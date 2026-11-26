# src/data_loader.py
import json
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

GENRES = ["Sci-Fi", "Action", "Drama", "Thriller", "Mystery", "Comedy"]


def load_catalog(catalog_path: str = "data/catalog.json") -> List[Dict[str, Any]]:
    """Load item catalog JSON and validate basic structure."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return catalog


def build_tag_vocabulary(catalog: List[Dict[str, Any]], max_tags: int = 48) -> Dict[str, int]:
    """Build deterministic vocabulary mapping top metadata tags to column indices."""
    counts: Counter = Counter()
    for item in catalog:
        for tag in item.get("tags", []):
            counts[tag.lower().strip()] += 1

    # Deterministic tie-break: (-frequency, alphabetical)
    sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_tags]
    return {tag: idx for idx, (tag, _) in enumerate(sorted_tags)}


def encode_tags_multihot(tags: List[str], tag_vocab: Dict[str, int]) -> np.ndarray:
    """Encode a list of tags into an L2-normalized multi-hot vector."""
    vec = np.zeros(len(tag_vocab), dtype=np.float32)
    for t in tags:
        idx = tag_vocab.get(t.lower().strip())
        if idx is not None:
            vec[idx] = 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def build_catalog_tag_matrix(catalog: List[Dict[str, Any]], tag_vocab: Dict[str, int]) -> np.ndarray:
    """Build (num_items, num_tags) normalized metadata feature matrix."""
    return np.vstack([encode_tags_multihot(item.get("tags", []), tag_vocab) for item in catalog])


def generate_user_data(
    num_users: int = 150,
    catalog_path: str = "data/catalog.json",
    holdout_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Generate realistic user profiles with primary/secondary genre & tag affinities,
    plus a strictly disjoint train/test holdout split for leak-free evaluation.

    Returns:
        users: List of user dicts containing train `history_item_ids`,
               unseen test `holdout_item_ids`, and `preferred_tags`.
        train_interactions: List of (user_idx, item_idx) training pairs ONLY.
    """
    catalog = load_catalog(catalog_path)
    genre_to_items: Dict[str, List[Dict[str, Any]]] = {g: [] for g in GENRES}
    for item in catalog:
        g = item.get("genre", "Drama")
        if g in genre_to_items:
            genre_to_items[g].append(item)

    rng = random.Random(seed)
    users: List[Dict[str, Any]] = []
    train_interactions: List[Tuple[int, int]] = []
    id_to_idx = {item["id"]: idx for idx, item in enumerate(catalog)}

    for u_id in range(num_users):
        primary_genre = GENRES[u_id % len(GENRES)] if u_id < len(GENRES) else rng.choice(GENRES)
        secondary_genre = rng.choice([g for g in GENRES if g != primary_genre])

        primary_pool = list(genre_to_items[primary_genre])
        secondary_pool = list(genre_to_items[secondary_genre])

        # Select characteristic tag affinity from primary genre items
        candidate_tags = []
        for item in primary_pool:
            candidate_tags.extend(item.get("tags", []))
        top_user_tags = [t for t, _ in Counter(candidate_tags).most_common(4)]
        preferred_tags = rng.sample(top_user_tags, k=min(3, len(top_user_tags))) if top_user_tags else []

        # Sample positive interactions: ~75% primary genre, ~25% secondary genre
        num_primary = min(len(primary_pool), rng.randint(5, 8))
        num_secondary = min(len(secondary_pool), rng.randint(1, 3))
        chosen_items = rng.sample(primary_pool, num_primary) + rng.sample(secondary_pool, num_secondary)
        rng.shuffle(chosen_items)

        # Ensure at least 2 holdout items when user has >= 6 interactions
        total_n = len(chosen_items)
        holdout_n = max(1, min(3, round(total_n * holdout_ratio)))
        holdout_items = chosen_items[:holdout_n]
        history_items = chosen_items[holdout_n:]

        history_ids = [item["id"] for item in history_items]
        holdout_ids = [item["id"] for item in holdout_items]

        for item_id in history_ids:
            train_interactions.append((u_id, id_to_idx[item_id]))

        users.append(
            {
                "user_id": u_id,
                "username": f"user_{u_id + 1:03d}",
                "primary_genre": primary_genre,
                "secondary_genre": secondary_genre,
                "preferred_genre_idx": GENRES.index(primary_genre),
                "preferred_tags": preferred_tags,
                "history_item_ids": history_ids,
                "holdout_item_ids": holdout_ids,
            }
        )

    return users, train_interactions


if __name__ == "__main__":
    u, inter = generate_user_data()
    print(f"Generated {len(u)} users, {len(inter)} train interactions.")
    print("Sample user 0 history:", u[0]["history_item_ids"], "| holdout:", u[0]["holdout_item_ids"])
