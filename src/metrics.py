# src/metrics.py
import time
from typing import Any, Callable, Dict, List

import numpy as np


def evaluate_ranking_strategy(
    users: List[Dict[str, Any]],
    rank_fn: Callable[[Dict[str, Any], int], List[int]],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a candidate retrieval function strictly against unseen holdout targets.

    Args:
        users: User dicts containing `history_item_ids` and unseen `holdout_item_ids`.
        rank_fn: Callable `(user_dict, k) -> List[int]` returning top-k unseen item IDs.
        k: Primary cutoff rank (also logs HitRate@5 and NDCG@5).
    """
    hit_at_5: List[float] = []
    hit_at_k: List[float] = []
    recalls: List[float] = []
    ndcg_at_5: List[float] = []
    ndcg_at_k: List[float] = []
    mrrs: List[float] = []
    latencies: List[float] = []

    for user in users:
        holdout = set(user.get("holdout_item_ids", []))
        if not holdout:
            continue

        t0 = time.perf_counter()
        ranked_ids = rank_fn(user, k)[:k]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        top5 = ranked_ids[:5]
        hits_5 = [iid for iid in top5 if iid in holdout]
        hits_k = [iid for iid in ranked_ids if iid in holdout]

        hit_at_5.append(1.0 if hits_5 else 0.0)
        hit_at_k.append(1.0 if hits_k else 0.0)
        recalls.append(len(hits_k) / len(holdout))

        dcg5 = sum(1.0 / np.log2(r + 1) for r, iid in enumerate(top5, start=1) if iid in holdout)
        idcg5 = sum(1.0 / np.log2(i + 1) for i in range(1, min(5, len(holdout)) + 1))
        ndcg_at_5.append(dcg5 / max(1e-8, idcg5))

        dcgk = sum(1.0 / np.log2(r + 1) for r, iid in enumerate(ranked_ids, start=1) if iid in holdout)
        idcgk = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(holdout)) + 1))
        ndcg_at_k.append(dcgk / max(1e-8, idcgk))

        rr = 0.0
        for rank, iid in enumerate(ranked_ids, start=1):
            if iid in holdout:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    return {
        "evaluated_users": len(hit_at_k),
        "HitRate@5": round(float(np.mean(hit_at_5)), 4),
        f"HitRate@{k}": round(float(np.mean(hit_at_k)), 4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        "NDCG@5": round(float(np.mean(ndcg_at_5)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcg_at_k)), 4),
        f"MRR@{k}": round(float(np.mean(mrrs)), 4),
        "P50_Latency_ms": round(float(np.percentile(latencies, 50)), 4),
        "P95_Latency_ms": round(float(np.percentile(latencies, 95)), 4),
    }
