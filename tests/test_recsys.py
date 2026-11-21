# tests/test_recsys.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
from src.model import TwoTowerNumPyEngine
from src.data_loader import generate_user_data
from src.engine import RecommendationEngine

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

def test_dual_tower_embeddings_norm():
    engine = TwoTowerNumPyEngine(num_users=10, num_items=15, embed_dim=16)
    u_vecs = [engine.get_user_vector(i) for i in range(10)]
    i_vecs = engine.get_all_item_vectors()

    # Verify unit length L2 norm for cosine dot product
    for u in u_vecs:
        assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-4)
    for i in i_vecs:
        assert np.isclose(np.linalg.norm(i), 1.0, atol=1e-4)

def test_candidate_generation():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=16)
    engine.initialize()

    res = engine.recommend_for_user(user_idx=0, top_k=4, exclude_watched=True)
    assert len(res["recommendations"]) == 4
    assert res["latency_ms"] >= 0.0

    # Ensure recommended items are not in user watch history
    watched_ids = {m["id"] for m in res["watch_history"]}
    for rec in res["recommendations"]:
        assert rec["id"] not in watched_ids

def test_evaluation_metrics():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=16)
    engine.initialize()

    metrics = engine.evaluate_metrics(k=5)
    assert "HitRate@5" in metrics
    assert "Recall@5" in metrics
    assert "NDCG@5" in metrics
    assert metrics["Recall@5"] >= 0.0
