# benchmarks/eval_movielens.py
"""
Real-World Dataset Validation on MovieLens-100K (GroupLens).
Trains Two-Tower Neural Retrieval (InfoNCE + LogQ) on historical ratings (>= 4)
with strict chronological Leave-Last-2-Out holdout evaluation and FAISS indexing.
"""
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.metrics import evaluate_ranking_strategy
from src.vector_index import create_vector_index

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_movielens_100k(data_dir: str) -> str:
    ml_dir = os.path.join(data_dir, "ml-100k")
    if os.path.exists(os.path.join(ml_dir, "u.data")) and os.path.exists(os.path.join(ml_dir, "u.item")):
        return ml_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens-100K from {ML100K_URL}...")
        resp = requests.get(ML100K_URL, timeout=30)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    return ml_dir


def load_movielens_split(ml_dir: str) -> Tuple[int, int, np.ndarray, List[Dict], List[Tuple[int, int]]]:
    """Load MovieLens-100K positive ratings (>= 4) with chronological Leave-Last-2-Out split."""
    item_genres_list: List[np.ndarray] = []
    item_id_map: Dict[int, int] = {}

    with open(os.path.join(ml_dir, "u.item"), "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            raw_id = int(parts[0])
            genre_flags = np.array([float(x) for x in parts[-19:]], dtype=np.float32)
            idx = len(item_genres_list)
            item_id_map[raw_id] = idx
            item_genres_list.append(genre_flags)

    item_genre_matrix = np.vstack(item_genres_list)
    num_items = len(item_genre_matrix)

    user_events = defaultdict(list)
    with open(os.path.join(ml_dir, "u.data"), "r", encoding="utf-8") as f:
        for line in f:
            u_raw, i_raw, rating, ts = line.strip().split("\t")
            if int(rating) >= 4 and int(i_raw) in item_id_map:
                user_events[int(u_raw)].append((int(ts), item_id_map[int(i_raw)]))

    users: List[Dict] = []
    train_interactions: List[Tuple[int, int]] = []
    user_genre_profiles: List[np.ndarray] = []

    for u_raw in sorted(user_events.keys()):
        events = sorted(user_events[u_raw], key=lambda x: x[0])
        if len(events) < 6:
            continue

        u_idx = len(users)
        holdout_ids = [iid for _, iid in events[-2:]]
        history_ids = [iid for _, iid in events[:-2]]

        g_prof = np.mean(item_genre_matrix[history_ids], axis=0)
        g_prof /= max(float(np.linalg.norm(g_prof)), 1e-8)
        user_genre_profiles.append(g_prof)

        for iid in history_ids:
            train_interactions.append((u_idx, iid))

        users.append(
            {
                "user_id": u_idx,
                "raw_user_id": u_raw,
                "history_item_ids": history_ids,
                "holdout_item_ids": holdout_ids,
            }
        )

    return len(users), num_items, item_genre_matrix, users, train_interactions, np.vstack(user_genre_profiles)


class MovieLensTwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_genres: int = 19, embed_dim: int = 48, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        self.u_emb = nn.Embedding(num_users, embed_dim)
        self.ug_proj = nn.Linear(num_genres, embed_dim)
        self.u_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, embed_dim),
        )

        self.i_emb = nn.Embedding(num_items, embed_dim)
        self.ig_proj = nn.Linear(num_genres, embed_dim)
        self.i_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, embed_dim),
        )

    def encode_users(self, u_ids: torch.Tensor, u_genres: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.u_emb(u_ids), self.ug_proj(u_genres)], dim=-1)
        return F.normalize(self.u_mlp(x), p=2, dim=-1)

    def encode_items(self, i_ids: torch.Tensor, i_genres: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.i_emb(i_ids), self.ig_proj(i_genres)], dim=-1)
        return F.normalize(self.i_mlp(x), p=2, dim=-1)

    def forward(self, u_ids, u_genres, i_ids, i_genres):
        u = self.encode_users(u_ids, u_genres)
        v = self.encode_items(i_ids, i_genres)
        return torch.matmul(u, v.T) / self.tau


def run_movielens_eval(epochs: int = 18, batch_size: int = 256, lr: float = 0.004) -> Dict:
    torch.manual_seed(42)
    np.random.seed(42)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_dir = ensure_movielens_100k(os.path.join(root_dir, "data"))
    num_users, num_items, item_genres, users, train_pairs, user_genres = load_movielens_split(ml_dir)

    print("=" * 84)
    print(f"REAL DATASET BENCHMARK: MOVIELENS-100K (Users={num_users}, Items={num_items}, Train={len(train_pairs)})")
    print("=" * 84)

    model = MovieLensTwoTower(num_users=num_users, num_items=num_items, num_genres=19, embed_dim=48, tau=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    u_ids_t = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    i_ids_t = torch.tensor([i for _, i in train_pairs], dtype=torch.long)
    u_genres_t = torch.tensor(user_genres, dtype=torch.float32)
    i_genres_t = torch.tensor(item_genres, dtype=torch.float32)

    item_counts = np.bincount([i for _, i in train_pairs], minlength=num_items).astype(np.float32)
    log_q_all = torch.tensor(np.log(np.maximum(item_counts, 1.0) / np.sum(item_counts)), dtype=torch.float32)

    indices = np.arange(len(train_pairs))
    model.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        ep_loss = []
        for s in range(0, len(indices), batch_size):
            b = indices[s : s + batch_size]
            if len(b) < 8:
                continue
            bu, bi = u_ids_t[b], i_ids_t[b]
            optimizer.zero_grad()
            sim = model(bu, u_genres_t[bu], bi, i_genres_t[bi])
            labels = torch.arange(len(b))
            loss_u2i = F.cross_entropy(sim - log_q_all[bi].view(1, -1), labels)
            loss_i2u = F.cross_entropy(sim.T, labels)
            loss = 0.5 * (loss_u2i + loss_i2u)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        if ep in (1, 6, 12, epochs):
            print(f"Epoch [{ep:02d}/{epochs:02d}] — LogQ InfoNCE Loss: {np.mean(ep_loss):.4f}")

    model.eval()
    with torch.no_grad():
        all_u_vecs = model.encode_users(torch.arange(num_users), u_genres_t).numpy()
        all_i_vecs = model.encode_items(torch.arange(num_items), i_genres_t).numpy()

    flat_idx = create_vector_index("flat", embed_dim=48)
    flat_idx.build(all_i_vecs, item_ids=list(range(num_items)))

    hnsw_idx = create_vector_index("hnsw", embed_dim=48)
    hnsw_idx.build(all_i_vecs, item_ids=list(range(num_items)))

    pop_counts = Counter(i for _, i in train_pairs)
    pop_ranked = [i for i, _ in pop_counts.most_common()] + [i for i in range(num_items) if i not in pop_counts]

    rng = np.random.default_rng(42)

    def random_ranker(u, top_k):
        hist = set(u["history_item_ids"])
        cands = [i for i in range(num_items) if i not in hist]
        return [int(x) for x in rng.choice(cands, size=min(top_k, len(cands)), replace=False)]

    def pop_ranker(u, top_k):
        hist = set(u["history_item_ids"])
        return [i for i in pop_ranked if i not in hist][:top_k]

    def make_faiss_ranker(vec_index):
        def _ranker(u, top_k):
            hist = set(u["history_item_ids"])
            raw = vec_index.search(all_u_vecs[u["user_id"]], top_k=min(num_items, top_k + len(hist) + 5))
            return [iid for iid, _ in raw if iid not in hist][:top_k]
        return _ranker

    models = [
        ("Random Baseline", random_ranker),
        ("Global Popularity", pop_ranker),
        (f"Two-Tower ({hnsw_idx.backend_name.upper()})", make_faiss_ranker(hnsw_idx)),
        (f"Two-Tower ({flat_idx.backend_name.upper()})", make_faiss_ranker(flat_idx)),
    ]

    summary = []
    print("-" * 84)
    print(f"{'MovieLens-100K Retriever':<32} | {'Hit@5':<7} | {'Hit@10':<7} | {'NDCG@10':<7} | {'MRR@10':<7} | {'P95 (ms)':<8}")
    print("-" * 84)
    for name, fn in models:
        m = evaluate_ranking_strategy(users, fn, k=10)
        m["model"] = name
        summary.append(m)
        print(
            f"{name:<32} | {m['HitRate@5']:<7.4f} | {m['HitRate@10']:<7.4f} | "
            f"{m['NDCG@10']:<7.4f} | {m['MRR@10']:<7.4f} | {m['P95_Latency_ms']:<8.4f}"
        )
    print("=" * 84)

    out_path = os.path.join(root_dir, "benchmarks", "movielens_results.json")
    payload = {
        "dataset": "MovieLens-100K (GroupLens)",
        "split": "Chronological Leave-Last-2-Out (positive ratings >= 4)",
        "num_users": num_users,
        "num_items": num_items,
        "train_interactions": len(train_pairs),
        "results": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved MovieLens-100K benchmark to {out_path}")
    _plot_movielens_chart(summary, os.path.join(root_dir, "assets", "movielens_benchmark.png"))
    return payload


def _plot_movielens_chart(results: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels = [r["model"].replace("Two-Tower ", "TT ") for r in results]
    hit5 = [r["HitRate@5"] for r in results]
    hit10 = [r["HitRate@10"] for r in results]
    ndcg10 = [r["NDCG@10"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#111114")

    b1 = ax.bar(x - width, hit5, width, label="HitRate@5", color="#fb923c", alpha=0.92)
    b2 = ax.bar(x, hit10, width, label="HitRate@10", color="#38bdf8", alpha=0.92)
    b3 = ax.bar(x + width, ndcg10, width, label="NDCG@10", color="#34d399", alpha=0.92)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#e4e4e7",
            )

    ax.set_title(
        "MovieLens-100K Validation (934 Users · 1,682 Items · Chronological Holdout)",
        color="#fafafa",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score (Higher is Better)", color="#a1a1aa", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e4e4e7", fontsize=8.5, rotation=8)
    ax.set_ylim(0, max(hit10) * 1.24)
    ax.tick_params(colors="#a1a1aa")
    ax.grid(axis="y", color="#27272a", linestyle="--", alpha=0.7)
    ax.legend(facecolor="#18181b", edgecolor="#27272a", labelcolor="#fafafa", loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved MovieLens chart to {out_path}")


if __name__ == "__main__":
    run_movielens_eval()
