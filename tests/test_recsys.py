# tests/test_recsys.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from src.data_loader import GENRES, generate_user_data, load_catalog
from src.engine import RecommendationEngine
from src.model import TwoTowerNumPyEngine
from src.vector_index import create_vector_index, project_embeddings_2d

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")


def test_catalog_and_disjoint_holdout_split():
    catalog = load_catalog(CATALOG_PATH)
    assert len(catalog) >= 50

    users, train_interactions = generate_user_data(num_users=50, catalog_path=CATALOG_PATH)
    assert len(users) == 50
    assert len(train_interactions) > 0

    for u in users:
        hist = set(u["history_item_ids"])
        holdout = set(u["holdout_item_ids"])
        # Zero train/test leakage between watch history and evaluation holdout targets
        assert len(hist) > 0
        assert len(holdout) > 0
        assert hist.isdisjoint(holdout)


def test_dual_tower_embeddings_norm():
    engine = TwoTowerNumPyEngine(num_users=10, num_items=15, embed_dim=16)
    u_vecs = [engine.get_user_vector(i) for i in range(10)]
    i_vecs = engine.get_all_item_vectors()

    for u in u_vecs:
        assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-4)
    for i in i_vecs:
        assert np.isclose(np.linalg.norm(i), 1.0, atol=1e-4)


def test_vector_index_backends_and_2d_projection():
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((60, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    item_ids = list(range(101, 161))
    query = vecs[0]

    for idx_type in ("flat", "hnsw", "ivf"):
        index = create_vector_index(idx_type, embed_dim=32)
        index.build(vecs, item_ids=item_ids)
        hits = index.search(query, top_k=5)
        assert len(hits) == 5
        assert hits[0][0] == 101
        assert np.isclose(hits[0][1], 1.0, atol=1e-3)

    coords = project_embeddings_2d(vecs)
    assert coords.shape == (60, 2)
    assert np.max(np.abs(coords)) <= 1.0 + 1e-4


def test_candidate_generation_and_cold_start():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=32, num_users=150)
    engine.initialize()

    res = engine.recommend_for_user(user_idx=0, top_k=6, exclude_watched=True, index_type="hnsw")
    assert len(res["recommendations"]) == 6
    assert res["latency_ms"] >= 0.0
    assert "ann_search_ms" in res["latency_breakdown"]

    watched_ids = {m["id"] for m in res["watch_history"]}
    for rec in res["recommendations"]:
        assert rec["id"] not in watched_ids

    cold_user = engine.recommend_cold_start(preferred_genre="Sci-Fi", tags=["space", "ai"], top_k=5)
    assert cold_user["cold_start"] is True
    assert len(cold_user["recommendations"]) == 5

    cold_item = engine.match_cold_start_item(title="Quantum Horizon", genre="Sci-Fi", tags=["time", "space"], top_k=4)
    assert len(cold_item["similar_catalog_items"]) == 4


def test_leak_free_evaluation_metrics():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=32, num_users=150)
    engine.initialize()

    metrics = engine.evaluate_metrics(k=10, index_type="flat")
    assert metrics["evaluated_users"] == 150
    assert metrics["HitRate@10"] > 0.35
    assert metrics["NDCG@10"] > 0.15
    assert metrics["P95_Latency_ms"] >= 0.0


def test_fastapi_endpoints_e2e():
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_json = health_resp.json()
        assert health_json["status"] == "HEALTHY"
        assert health_json["total_items"] == 60
        assert "hnsw" in health_json["vector_indexes"]

        users_resp = client.get("/api/v1/users")
        assert users_resp.status_code == 200
        assert len(users_resp.json()) == 150

        rec_resp = client.post("/api/v1/recommend", json={"user_id": 0, "top_k": 5, "index_type": "hnsw"})
        assert rec_resp.status_code == 200
        assert len(rec_resp.json()["recommendations"]) == 5

        cold_u = client.post("/api/v1/cold-start/user", json={"preferred_genre": "Thriller", "tags": ["detective", "noir"], "top_k": 4})
        assert cold_u.status_code == 200
        assert len(cold_u.json()["recommendations"]) == 4

        cold_i = client.post("/api/v1/cold-start/item", json={"title": "Cipher Zero", "genre": "Mystery", "tags": ["twist", "detective"], "top_k": 4})
        assert cold_i.status_code == 200
        assert len(cold_i.json()["similar_catalog_items"]) == 4

        map_resp = client.get("/api/v1/latent-map")
        assert map_resp.status_code == 200
        assert map_resp.json()["total_items"] == 60

        metrics_resp = client.get("/api/v1/metrics?k=10&index_type=flat")
        assert metrics_resp.status_code == 200
        assert "NDCG@10" in metrics_resp.json()
