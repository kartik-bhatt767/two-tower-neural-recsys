# tests/test_recsys.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.data_loader import generate_user_data, load_catalog
from src.engine import RecommendationEngine
from src.model import TwoTowerNumPyEngine
from src.vector_index import create_vector_index, project_embeddings_2d_with_basis

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

    coords, mean_vec, basis, scale = project_embeddings_2d_with_basis(vecs)
    assert coords.shape == (60, 2)
    assert mean_vec.shape == (32,)
    assert basis.shape == (2, 32)
    assert scale > 0.0
    assert np.max(np.abs(coords)) <= 1.0 + 1e-4


def test_candidate_generation_stage2_bounds_and_title_cold_start():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=32, num_users=150)
    engine.initialize()

    res = engine.recommend_for_user(
        user_idx=0, top_k=6, exclude_watched=True, index_type="hnsw", use_stage2=True, diversity_lambda=0.88
    )
    assert len(res["recommendations"]) == 6
    assert res["stage2_enabled"] is True
    assert "stage2_rerank_ms" in res["latency_breakdown"]
    assert "stage1_rank" in res["recommendations"][0]
    assert "rank_delta" in res["recommendations"][0]

    watched_ids = {m["id"] for m in res["watch_history"]}
    for rec in res["recommendations"]:
        assert rec["id"] not in watched_ids

    with pytest.raises(KeyError):
        engine.recommend_for_user(user_idx=9999)

    sim = engine.recommend_similar_items(item_id=1, top_k=4)
    assert sim["source_item"]["id"] == 1
    assert len(sim["similar_items"]) == 4
    assert all(item["id"] != 1 for item in sim["similar_items"])

    cold_user = engine.recommend_cold_start(preferred_genre="Sci-Fi", tags=["space", "ai"], top_k=5)
    assert cold_user["cold_start"] is True
    assert len(cold_user["recommendations"]) == 5

    # Distinct unseen item titles under same genre/tags should steer matching neighbors
    cold_mars = engine.match_cold_start_item(title="Mars Engineering Survival", genre="Sci-Fi", tags=["time"], top_k=3)
    cold_cyber = engine.match_cold_start_item(title="Cyberpunk AI Hacker Simulation", genre="Sci-Fi", tags=["time"], top_k=3)
    mars_titles = [m["title"] for m in cold_mars["similar_catalog_items"]]
    cyber_titles = [m["title"] for m in cold_cyber["similar_catalog_items"]]
    assert "The Martian" in mars_titles
    assert "The Matrix" in cyber_titles
    assert mars_titles != cyber_titles


def test_leak_free_evaluation_metrics():
    engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=32, num_users=150)
    engine.initialize()

    metrics_s1 = engine.evaluate_metrics(k=10, index_type="flat", use_stage2=False)
    metrics_s2 = engine.evaluate_metrics(k=10, index_type="hnsw", use_stage2=True)
    assert metrics_s1["evaluated_users"] == 150
    assert metrics_s1["HitRate@10"] > 0.35
    assert metrics_s2["HitRate@10"] > 0.35
    assert metrics_s2["stage2_enabled"] is True


def test_fastapi_endpoints_e2e_and_error_handling():
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

        cat_resp = client.get("/api/v1/catalog?genre=Sci-Fi&q=mars")
        assert cat_resp.status_code == 200
        assert cat_resp.json()["total"] == 1
        assert cat_resp.json()["items"][0]["title"] == "The Martian"

        cat_page = client.get("/api/v1/catalog?limit=10&offset=10")
        assert cat_page.status_code == 200
        assert len(cat_page.json()["items"]) == 10
        assert cat_page.json()["total"] == 60

        rec_resp = client.post(
            "/api/v1/recommend",
            json={"user_id": 0, "top_k": 5, "index_type": "hnsw", "use_stage2": True, "diversity_lambda": 0.85},
        )
        assert rec_resp.status_code == 200
        assert len(rec_resp.json()["recommendations"]) == 5

        bad_user_resp = client.post("/api/v1/recommend", json={"user_id": 9999, "top_k": 5})
        assert bad_user_resp.status_code == 404

        sim_resp = client.get("/api/v1/similar-items/1?top_k=4")
        assert sim_resp.status_code == 200
        assert len(sim_resp.json()["similar_items"]) == 4

        bad_sim = client.get("/api/v1/similar-items/9999")
        assert bad_sim.status_code == 404

        cold_u = client.post("/api/v1/cold-start/user", json={"preferred_genre": "Thriller", "tags": ["detective", "noir"], "top_k": 4})
        assert cold_u.status_code == 200
        assert len(cold_u.json()["recommendations"]) == 4

        cold_i = client.post("/api/v1/cold-start/item", json={"title": "Cipher Zero", "genre": "Mystery", "tags": ["twist", "detective"], "top_k": 4})
        assert cold_i.status_code == 200
        assert len(cold_i.json()["similar_catalog_items"]) == 4

        map_resp = client.get("/api/v1/latent-map")
        assert map_resp.status_code == 200
        assert map_resp.json()["total_items"] == 60
        assert "pca_projection" in map_resp.json()

        metrics_resp = client.get("/api/v1/metrics?k=10&index_type=flat&use_stage2=true")
        assert metrics_resp.status_code == 200
        assert "NDCG@10" in metrics_resp.json()
