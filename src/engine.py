# src/engine.py
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data_loader import (
    GENRES,
    build_catalog_tag_matrix,
    build_tag_vocabulary,
    encode_tags_multihot,
    generate_user_data,
    load_catalog,
)
from src.model import TORCH_AVAILABLE, TwoTowerNumPyEngine, TwoTowerRecSys
from src.metrics import evaluate_ranking_strategy
from src.vector_index import BaseVectorIndex, create_vector_index, project_embeddings_2d

if TORCH_AVAILABLE:
    import torch


class RecommendationEngine:
    """
    Production Two-Tower Neural Candidate Retrieval Engine backed by FAISS Vector Indexing.
    Supports trained PyTorch dual-tower inference, cold-start metadata encoding,
    Exact FlatIP / HNSW / IVF-Flat ANN search, and leak-free holdout evaluation.
    """

    def __init__(
        self,
        catalog_path: str = "data/catalog.json",
        weights_path: Optional[str] = None,
        embed_dim: int = 32,
        num_users: int = 150,
    ):
        self.catalog_path = catalog_path
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.abspath(catalog_path)), "..", "models", "two_tower_weights.pt"
            )
        self.weights_path = weights_path
        self.embed_dim = embed_dim
        self.num_users = num_users

        self.catalog: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.train_interactions: List[Tuple[int, int]] = []
        self.item_id_map: Dict[int, Dict[str, Any]] = {}
        self.id_to_pos: Dict[int, int] = {}
        self.tag_vocab: Dict[str, int] = {}
        self.item_tag_matrix: Optional[np.ndarray] = None
        self.user_tag_matrix: Optional[np.ndarray] = None

        self.torch_model: Optional["TwoTowerRecSys"] = None
        self.numpy_engine: Optional[TwoTowerNumPyEngine] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None

        self.indexes: Dict[str, BaseVectorIndex] = {}
        self.default_index_type: str = "hnsw"
        self.inference_mode: str = "numpy_fallback"

    def initialize(self) -> None:
        """Load catalog, split train/holdout users, encode dual towers, and build MIPS vector indexes."""
        self.catalog = load_catalog(self.catalog_path)
        self.item_id_map = {item["id"]: item for item in self.catalog}
        self.id_to_pos = {item["id"]: idx for idx, item in enumerate(self.catalog)}
        self.tag_vocab = build_tag_vocabulary(self.catalog)
        self.item_tag_matrix = build_catalog_tag_matrix(self.catalog, self.tag_vocab)

        self.users, self.train_interactions = generate_user_data(
            num_users=self.num_users,
            catalog_path=self.catalog_path,
            holdout_ratio=0.25,
            seed=42,
        )
        self.user_tag_matrix = np.vstack(
            [encode_tags_multihot(u.get("preferred_tags", []), self.tag_vocab) for u in self.users]
        )

        loaded_torch = self._try_load_pytorch_checkpoint()
        if not loaded_torch:
            self._initialize_numpy_engine()

        # Build Exact FlatIP, HNSW graph ANN, and Voronoi IVF-Flat vector indexes
        item_ids = [item["id"] for item in self.catalog]
        for idx_name in ("flat", "hnsw", "ivf"):
            vec_index = create_vector_index(idx_name, embed_dim=self.embed_dim)
            vec_index.build(self.item_embeddings, item_ids=item_ids)
            self.indexes[idx_name] = vec_index

    def _try_load_pytorch_checkpoint(self) -> bool:
        """Load pretrained PyTorch Two-Tower model weights when compatible with catalog."""
        if not TORCH_AVAILABLE or not os.path.exists(self.weights_path):
            return False

        try:
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                return False

            if (
                ckpt.get("num_users") != len(self.users)
                or ckpt.get("num_items") != len(self.catalog)
                or ckpt.get("num_genres") != len(GENRES)
                or ckpt.get("num_tags") != len(self.tag_vocab)
            ):
                return False

            model = TwoTowerRecSys(
                num_users=ckpt["num_users"],
                num_items=ckpt["num_items"],
                num_genres=ckpt["num_genres"],
                num_tags=ckpt["num_tags"],
                embed_dim=ckpt["embed_dim"],
                temperature=ckpt.get("temperature", 0.07),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.torch_model = model
            self.embed_dim = ckpt["embed_dim"]
            self.inference_mode = "pytorch_neural"

            with torch.no_grad():
                u_ids = torch.arange(len(self.users), dtype=torch.long)
                u_genres = torch.tensor([u["preferred_genre_idx"] for u in self.users], dtype=torch.long)
                u_tags = torch.tensor(self.user_tag_matrix, dtype=torch.float32)
                self.user_embeddings = model.user_tower(u_ids, u_genres, u_tags).cpu().numpy()

                i_ids = torch.arange(len(self.catalog), dtype=torch.long)
                i_genres = torch.tensor(
                    [GENRES.index(item["genre"]) if item["genre"] in GENRES else 0 for item in self.catalog],
                    dtype=torch.long,
                )
                i_tags = torch.tensor(self.item_tag_matrix, dtype=torch.float32)
                self.item_embeddings = model.item_tower(i_ids, i_genres, i_tags).cpu().numpy()

            return True
        except Exception:
            return False

    def _initialize_numpy_engine(self) -> None:
        """Train metadata-aware NumPy dual-tower fallback strictly on train interactions."""
        num_items = len(self.catalog)
        num_users = len(self.users)
        engine = TwoTowerNumPyEngine(num_users=num_users, num_items=num_items, embed_dim=self.embed_dim)

        user_genres = [u["preferred_genre_idx"] for u in self.users]
        item_genres = [GENRES.index(item["genre"]) if item["genre"] in GENRES else 0 for item in self.catalog]
        engine.inject_metadata_priors(
            user_genre_indices=user_genres,
            item_genre_indices=item_genres,
            num_genres=len(GENRES),
            user_tag_matrix=self.user_tag_matrix,
            item_tag_matrix=self.item_tag_matrix,
        )
        engine.fit_interactions(self.train_interactions, epochs=25, lr=0.08)

        self.numpy_engine = engine
        self.user_embeddings = engine.user_embeddings.copy()
        self.item_embeddings = engine.item_embeddings.copy()
        self.inference_mode = "numpy_dual_tower"

    def _encode_online_user(self, user: Dict[str, Any]) -> np.ndarray:
        """Compute active user tower embedding cleanly at inference time."""
        if self.torch_model is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                u_id = torch.tensor([user["user_id"]], dtype=torch.long)
                u_genre = torch.tensor([user["preferred_genre_idx"]], dtype=torch.long)
                u_tags = torch.tensor(
                    self.user_tag_matrix[user["user_id"]].reshape(1, -1), dtype=torch.float32
                )
                return self.torch_model.user_tower(u_id, u_genre, u_tags).cpu().numpy()[0]
        return self.user_embeddings[user["user_id"]]

    def _resolve_index(self, index_type: Optional[str]) -> BaseVectorIndex:
        key = (index_type or self.default_index_type).lower().strip()
        if key not in self.indexes:
            raise ValueError(f"Unknown index_type '{key}'. Choose from {list(self.indexes.keys())}.")
        return self.indexes[key]

    @staticmethod
    def _format_candidate(movie: Dict[str, Any], score: float) -> Dict[str, Any]:
        # Calibrate cosine dot-product [-1, 1] into intuitive match percentage [0, 99]
        affinity_pct = int(min(99, max(5, round((float(score) + 1.0) * 50.0))))
        return {
            "id": movie["id"],
            "title": movie["title"],
            "genre": movie["genre"],
            "year": movie["year"],
            "rating": movie.get("rating", 8.0),
            "director": movie.get("director", "Unknown"),
            "tags": movie["tags"],
            "description": movie.get("description", ""),
            "similarity_score": round(float(score), 4),
            "affinity_pct": affinity_pct,
        }

    def recommend_for_user(
        self,
        user_idx: int,
        top_k: int = 6,
        exclude_watched: bool = True,
        index_type: str = "hnsw",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        user = self.users[user_idx % len(self.users)]
        watched_set = set(user["history_item_ids"])
        vec_index = self._resolve_index(index_type)

        t_u0 = time.perf_counter()
        u_vec = self._encode_online_user(user)
        user_tower_ms = (time.perf_counter() - t_u0) * 1000.0

        # Retrieve extra headroom from ANN index to accommodate watched filtering
        fetch_k = min(len(self.catalog), top_k + (len(watched_set) if exclude_watched else 0) + 5)
        t_ann0 = time.perf_counter()
        ranked_candidates = vec_index.search(u_vec, top_k=fetch_k)
        ann_search_ms = (time.perf_counter() - t_ann0) * 1000.0

        t_f0 = time.perf_counter()
        recommendations = []
        for item_id, score in ranked_candidates:
            if exclude_watched and item_id in watched_set:
                continue
            movie = self.item_id_map[item_id]
            recommendations.append(self._format_candidate(movie, score))
            if len(recommendations) >= top_k:
                break
        filter_ms = (time.perf_counter() - t_f0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        history_movies = [self.item_id_map[m_id] for m_id in user["history_item_ids"] if m_id in self.item_id_map]

        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "preferred_genre": user["primary_genre"],
            "preferred_tags": user.get("preferred_tags", []),
            "inference_mode": self.inference_mode,
            "index_backend": vec_index.backend_name,
            "watch_history": history_movies,
            "recommendations": recommendations,
            "latency_ms": round(total_ms, 3),
            "latency_breakdown": {
                "user_tower_ms": round(user_tower_ms, 3),
                "ann_search_ms": round(ann_search_ms, 3),
                "filter_ms": round(filter_ms, 3),
                "total_ms": round(total_ms, 3),
            },
        }

    def recommend_cold_start(
        self,
        preferred_genre: str,
        tags: List[str],
        top_k: int = 6,
        index_type: str = "flat",
    ) -> Dict[str, Any]:
        """Generate recommendations for a brand-new cold-start user with zero watch history."""
        t0 = time.perf_counter()
        genre_idx = GENRES.index(preferred_genre) if preferred_genre in GENRES else 0
        tag_vec = encode_tags_multihot(tags, self.tag_vocab)
        vec_index = self._resolve_index(index_type)

        if self.torch_model is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                g_tensor = torch.tensor([genre_idx], dtype=torch.long)
                t_tensor = torch.tensor(tag_vec.reshape(1, -1), dtype=torch.float32)
                query_vec = self.torch_model.user_tower.encode_cold_user(g_tensor, t_tensor).cpu().numpy()[0]
        else:
            # Blend centroids of matching genre users & tag affinities
            matching_users = [u["user_id"] for u in self.users if u["preferred_genre_idx"] == genre_idx]
            base_vec = (
                np.mean(self.user_embeddings[matching_users], axis=0)
                if matching_users
                else np.mean(self.user_embeddings, axis=0)
            )
            matching_items = [
                idx for idx, item in enumerate(self.catalog) if any(t in item.get("tags", []) for t in tags)
            ]
            if matching_items:
                tag_vec_emb = np.mean(self.item_embeddings[matching_items], axis=0)
                query_vec = 0.65 * base_vec + 0.35 * tag_vec_emb
            else:
                query_vec = base_vec
            query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-8)

        ranked = vec_index.search(query_vec, top_k=top_k)
        recommendations = [self._format_candidate(self.item_id_map[item_id], score) for item_id, score in ranked]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "cold_start": True,
            "preferred_genre": preferred_genre,
            "input_tags": tags,
            "inference_mode": self.inference_mode,
            "index_backend": vec_index.backend_name,
            "recommendations": recommendations,
            "latency_ms": round(latency_ms, 3),
        }

    def match_cold_start_item(
        self, title: str, genre: str, tags: List[str], top_k: int = 5
    ) -> Dict[str, Any]:
        """Project a brand-new unseen catalog item from metadata and retrieve nearest items."""
        t0 = time.perf_counter()
        genre_idx = GENRES.index(genre) if genre in GENRES else 0
        tag_vec = encode_tags_multihot(tags, self.tag_vocab)

        if self.torch_model is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                g_t = torch.tensor([genre_idx], dtype=torch.long)
                t_t = torch.tensor(tag_vec.reshape(1, -1), dtype=torch.float32)
                item_vec = self.torch_model.item_tower.encode_cold_item(g_t, t_t).cpu().numpy()[0]
        else:
            same_genre_pos = [idx for idx, it in enumerate(self.catalog) if it["genre"] == genre]
            genre_vec = (
                np.mean(self.item_embeddings[same_genre_pos], axis=0)
                if same_genre_pos
                else np.mean(self.item_embeddings, axis=0)
            )
            tag_sim = self.item_tag_matrix @ tag_vec
            weighted = np.dot(tag_sim, self.item_embeddings)
            item_vec = 0.7 * genre_vec + 0.3 * weighted
            item_vec /= max(np.linalg.norm(item_vec), 1e-8)

        nearest = self.indexes["flat"].search(item_vec, top_k=top_k)
        similar_items = [self._format_candidate(self.item_id_map[item_id], score) for item_id, score in nearest]
        return {
            "title": title,
            "genre": genre,
            "tags": tags,
            "similar_catalog_items": similar_items,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def get_latent_space_map(self) -> Dict[str, Any]:
        """Project catalog item embeddings into 2D PCA coordinates for cluster visualization."""
        coords = project_embeddings_2d(self.item_embeddings)
        points = []
        for idx, item in enumerate(self.catalog):
            points.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "genre": item["genre"],
                    "year": item["year"],
                    "rating": item.get("rating", 8.0),
                    "x": round(float(coords[idx, 0]), 4),
                    "y": round(float(coords[idx, 1]), 4),
                }
            )
        return {"genres": GENRES, "total_items": len(points), "items": points}

    def evaluate_metrics(self, k: int = 10, index_type: str = "flat") -> Dict[str, Any]:
        """
        Strict leak-free evaluation on unseen `holdout_item_ids`.
        Training `history_item_ids` are masked out prior to Top-K cutoff ranking.
        """
        vec_index = self._resolve_index(index_type)

        def _ranker(user: Dict[str, Any], top_k: int) -> List[int]:
            history = set(user.get("history_item_ids", []))
            u_vec = self._encode_online_user(user)
            fetch_k = min(len(self.catalog), top_k + len(history) + 5)
            raw = vec_index.search(u_vec, top_k=fetch_k)
            return [item_id for item_id, _ in raw if item_id not in history][:top_k]

        metrics = evaluate_ranking_strategy(self.users, _ranker, k=k)
        metrics["index_backend"] = vec_index.backend_name
        metrics["inference_mode"] = self.inference_mode
        return metrics
