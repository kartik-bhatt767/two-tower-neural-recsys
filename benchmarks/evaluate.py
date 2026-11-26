# benchmarks/evaluate.py
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.engine import RecommendationEngine
from src.metrics import evaluate_ranking_strategy


def run_benchmark(k: int = 10) -> Dict[str, Any]:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    catalog_path = os.path.join(root_dir, "data", "catalog.json")

    engine = RecommendationEngine(catalog_path=catalog_path, embed_dim=32, num_users=150)
    engine.initialize()

    all_item_ids = [item["id"] for item in engine.catalog]
    pop_counts = Counter(engine.catalog[i_idx]["id"] for _, i_idx in engine.train_interactions)
    pop_ranked_ids = [iid for iid, _ in pop_counts.most_common()] + [
        iid for iid in all_item_ids if iid not in pop_counts
    ]

    rng = np.random.default_rng(42)

    def random_ranker(user: Dict[str, Any], top_k: int) -> List[int]:
        history = set(user.get("history_item_ids", []))
        candidates = [iid for iid in all_item_ids if iid not in history]
        chosen = rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
        return [int(x) for x in chosen]

    def popularity_ranker(user: Dict[str, Any], top_k: int) -> List[int]:
        history = set(user.get("history_item_ids", []))
        return [iid for iid in pop_ranked_ids if iid not in history][:top_k]

    def make_ann_ranker(index_type: str):
        vec_index = engine.indexes[index_type]

        def _ranker(user: Dict[str, Any], top_k: int) -> List[int]:
            history = set(user.get("history_item_ids", []))
            u_vec = engine._encode_online_user(user)
            fetch_k = min(len(all_item_ids), top_k + len(history) + 5)
            raw = vec_index.search(u_vec, top_k=fetch_k)
            return [iid for iid, _ in raw if iid not in history][:top_k]

        return _ranker

    models = [
        ("Random Baseline", "random", random_ranker),
        ("Global Popularity", "popularity", popularity_ranker),
        (
            f"Two-Tower ({engine.indexes['ivf'].backend_name.upper()})",
            engine.indexes["ivf"].backend_name,
            make_ann_ranker("ivf"),
        ),
        (
            f"Two-Tower ({engine.indexes['hnsw'].backend_name.upper()})",
            engine.indexes["hnsw"].backend_name,
            make_ann_ranker("hnsw"),
        ),
        (
            f"Two-Tower ({engine.indexes['flat'].backend_name.upper()})",
            engine.indexes["flat"].backend_name,
            make_ann_ranker("flat"),
        ),
    ]

    results: List[Dict[str, Any]] = []
    print("=" * 84)
    print(f"OFFLINE HOLDOUT EVALUATION BENCHMARK (K={k}, Mode={engine.inference_mode})")
    print(f"Catalog Size: {len(all_item_ids)} items | Users: {len(engine.users)} | Strict Zero-Leakage Holdout")
    print("=" * 84)
    print(f"{'Model / Retriever':<32} | {'Hit@5':<7} | {'Hit@10':<7} | {'NDCG@10':<7} | {'MRR@10':<7} | {'P95 (ms)':<8}")
    print("-" * 84)

    for display_name, backend_key, rank_fn in models:
        metrics = evaluate_ranking_strategy(engine.users, rank_fn, k=k)
        metrics["model"] = display_name
        metrics["backend"] = backend_key
        results.append(metrics)
        print(
            f"{display_name:<32} | "
            f"{metrics['HitRate@5']:<7.4f} | "
            f"{metrics[f'HitRate@{k}']:<7.4f} | "
            f"{metrics[f'NDCG@{k}']:<7.4f} | "
            f"{metrics[f'MRR@{k}']:<7.4f} | "
            f"{metrics['P95_Latency_ms']:<8.4f}"
        )

    print("=" * 84)

    out_json = os.path.join(root_dir, "benchmarks", "results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "k": k,
        "catalog_size": len(all_item_ids),
        "num_users": len(engine.users),
        "train_interactions": len(engine.train_interactions),
        "inference_mode": engine.inference_mode,
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_benchmark_chart(results, k, os.path.join(root_dir, "assets", "benchmark_comparison.png"))
    return payload


def _plot_benchmark_chart(results: List[Dict[str, Any]], k: int, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels = [r["model"].replace("Two-Tower ", "TT ") for r in results]
    ndcg = [r[f"NDCG@{k}"] for r in results]
    hit_k = [r[f"HitRate@{k}"] for r in results]
    mrr_k = [r[f"MRR@{k}"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    fig.patch.set_facecolor("#05070e")
    ax.set_facecolor("#0f172a")

    b1 = ax.bar(x - width, hit_k, width, label=f"HitRate@{k}", color="#f43f5e", alpha=0.92)
    b2 = ax.bar(x, ndcg, width, label=f"NDCG@{k}", color="#6366f1", alpha=0.92)
    b3 = ax.bar(x + width, mrr_k, width, label=f"MRR@{k}", color="#10b981", alpha=0.92)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#e2e8f0",
            )

    ax.set_title(
        f"Offline Holdout Benchmark: Two-Tower + FAISS vs. Baselines (Top-{k})",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score (Higher is Better)", color="#94a3b8", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e2e8f0", fontsize=8.5, rotation=10)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors="#cbd5e1")
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#1e293b", edgecolor="none", labelcolor="white", loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Benchmark chart saved to {out_path}")


if __name__ == "__main__":
    run_benchmark(k=10)
