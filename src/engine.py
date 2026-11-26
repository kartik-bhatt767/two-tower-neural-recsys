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
    extract_title_lexical_signals,
    generate_user_data,
    load_catalog,
)
from src.metrics import evaluate_ranking_strategy
from src.model import TORCH_AVAILABLE, TwoTowerNumPyEngine
from src.reranker import Stage2CrossFeatureReranker
from src.vector_index import (
    BaseVectorIndex,
    create_vector_index,
    project_embeddings_2d_with_basis,
)

if TORCH_AVAILABLE:
    import torch
    from src.model import TwoTowerRecSys
else:
    TwoTowerRecSys = None


class RecommendationEngine:
    """
    Production Two-Stage Neural Candidate Retrieval & Re-Ranking Engine backed by FAISS.
    Stage 1: Decoupled User/Item Dual Towers + FAISS HNSW/IVF-Flat/FlatIP MIPS search.
    Stage 2: Vectorized Cross-Feature Fine Re-Ranker + Maximal Marginal Relevance (MMR).
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
        self.reranker = Stage2CrossFeatureReranker()
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
    def _format_candidate(
        movie: Dict[str, Any],
        stage1_score: float,
        stage2_score: Optional[float] = None,
        stage1_rank: Optional[int] = None,
        final_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        blended = stage2_score if stage2_score is not None else (stage1_score + 1.0) * 0.5
        affinity_pct = int(min(99, max(5, round(float(blended) * 100.0))))
        rank_delta = (stage1_rank - final_rank) if (stage1_rank is not None and final_rank is not None) else 0
        return {
            "id": movie["id"],
            "title": movie["title"],
            "genre": movie["genre"],
            "year": movie["year"],
            "rating": movie.get("rating", 8.0),
            "director": movie.get("director", "Unknown"),
            "tags": movie["tags"],
            "description": movie.get("description", ""),
            "similarity_score": round(float(stage1_score), 4),
            "stage2_score": round(float(blended), 4),
            "stage1_rank": stage1_rank,
            "rank_delta": rank_delta,
            "affinity_pct": affinity_pct,
        }

    def recommend_for_user(
        self,
        user_idx: int,
        top_k: int = 6,
        exclude_watched: bool = True,
        index_type: str = "hnsw",
        use_stage2: bool = True,
        diversity_lambda: float = 0.88,
    ) -> Dict[str, Any]:
        if user_idx < 0 or user_idx >= len(self.users):
            raise KeyError(f"User ID {user_idx} not found (valid range: 0..{len(self.users) - 1}).")

        t0 = time.perf_counter()
        user = self.users[user_idx]
        watched_set = set(user["history_item_ids"])
        vec_index = self._resolve_index(index_type)
        history_movies = [
            self.item_id_map[m_id] for m_id in user["history_item_ids"] if m_id in self.item_id_map
        ]

        t_u0 = time.perf_counter()
        u_vec = self._encode_online_user(user)
        user_tower_ms = (time.perf_counter() - t_u0) * 1000.0

        stage1_pool = max(22, top_k * 3) if use_stage2 else top_k
        fetch_k = min(len(self.catalog), stage1_pool + (len(watched_set) if exclude_watched else 0) + 5)

        t_ann0 = time.perf_counter()
        ranked_candidates = vec_index.search(u_vec, top_k=fetch_k)
        ann_search_ms = (time.perf_counter() - t_ann0) * 1000.0

        t_f0 = time.perf_counter()
        stage1_filtered: List[Tuple[Dict[str, Any], float]] = []
        for item_id, score in ranked_candidates:
            if exclude_watched and item_id in watched_set:
                continue
            stage1_filtered.append((self.item_id_map[item_id], score))
            if not use_stage2 and len(stage1_filtered) >= top_k:
                break
        filter_ms = (time.perf_counter() - t_f0) * 1000.0

        stage2_rerank_ms = 0.0
        if use_stage2:
            t_r0 = time.perf_counter()
            reranked = self.reranker.rerank(
                user=user,
                candidates=stage1_filtered,
                item_embeddings=self.item_embeddings,
                id_to_pos=self.id_to_pos,
                top_k=top_k,
                diversity_lambda=diversity_lambda,
                history_movies=history_movies,
            )
            stage2_rerank_ms = (time.perf_counter() - t_r0) * 1000.0
            recommendations = [
                self._format_candidate(m, s1, s2, stage1_rank=r1, final_rank=rf)
                for m, s1, s2, r1, rf in reranked
            ]
        else:
            recommendations = [
                self._format_candidate(m, s1, stage1_rank=r1, final_rank=r1)
                for r1, (m, s1) in enumerate(stage1_filtered[:top_k], start=1)
            ]

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "preferred_genre": user["primary_genre"],
            "preferred_tags": user.get("preferred_tags", []),
            "inference_mode": self.inference_mode,
            "index_backend": vec_index.backend_name,
            "stage2_enabled": use_stage2,
            "diversity_lambda": diversity_lambda,
            "watch_history": history_movies,
            "recommendations": recommendations,
            "latency_ms": round(total_ms, 3),
            "latency_breakdown": {
                "user_tower_ms": round(user_tower_ms, 3),
                "ann_search_ms": round(ann_search_ms, 3),
                "filter_ms": round(filter_ms, 3),
                "stage2_rerank_ms": round(stage2_rerank_ms, 3),
                "total_ms": round(total_ms, 3),
            },
        }

    def recommend_similar_items(self, item_id: int, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve nearest item neighbors (Item-to-Item MIPS via FAISS FlatIP)."""
        if item_id not in self.id_to_pos:
            raise KeyError(f"Item ID {item_id} not found in catalog.")

        t0 = time.perf_counter()
        pos = self.id_to_pos[item_id]
        source_item = self.catalog[pos]
        item_vec = self.item_embeddings[pos]

        raw = self.indexes["flat"].search(item_vec, top_k=min(len(self.catalog), top_k + 1))
        similar = []
        for cand_id, score in raw:
            if cand_id == item_id:
                continue
            similar.append(self._format_candidate(self.item_id_map[cand_id], score))
            if len(similar) >= top_k:
                break

        return {
            "source_item": source_item,
            "similar_items": similar,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
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
        self, title: str, genre: str, tags: List[str], top_k: int = 6
    ) -> Dict[str, Any]:
        """Project a brand-new unseen catalog item from title + genre + tags and retrieve nearest items."""
        t0 = time.perf_counter()
        genre_idx = GENRES.index(genre) if genre in GENRES else 0
        merged_tags, lex_scores = extract_title_lexical_signals(title, tags, self.catalog, self.tag_vocab)
        tag_vec = encode_tags_multihot(merged_tags, self.tag_vocab)

        if self.torch_model is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                g_t = torch.tensor([genre_idx], dtype=torch.long)
                t_t = torch.tensor(tag_vec.reshape(1, -1), dtype=torch.float32)
                tower_vec = self.torch_model.item_tower.encode_cold_item(g_t, t_t).cpu().numpy()[0]
        else:
            same_genre_pos = [idx for idx, it in enumerate(self.catalog) if it["genre"] == genre]
            genre_vec = (
                np.mean(self.item_embeddings[same_genre_pos], axis=0)
                if same_genre_pos
                else np.mean(self.item_embeddings, axis=0)
            )
            tag_sim = self.item_tag_matrix @ tag_vec
            weighted = np.dot(tag_sim, self.item_embeddings)
            tower_vec = 0.65 * genre_vec + 0.35 * weighted
            tower_vec /= max(np.linalg.norm(tower_vec), 1e-8)

        # Blend neural Item Tower vector with semantic centroid of title-keyword matches
        if float(np.max(lex_scores)) > 0.0:
            lex_centroid = np.dot(lex_scores, self.item_embeddings)
            lex_centroid /= max(float(np.linalg.norm(lex_centroid)), 1e-8)
            item_vec = 0.55 * tower_vec + 0.45 * lex_centroid
            item_vec /= max(float(np.linalg.norm(item_vec)), 1e-8)
        else:
            item_vec = tower_vec

        # Fuse FAISS dot product with exact lexical match strength on title keywords
        raw_sims = (self.item_embeddings @ item_vec).astype(np.float32)
        blended_scores = 0.72 * ((raw_sims + 1.0) * 0.5) + 0.28 * lex_scores
        top_pos = np.argsort(blended_scores)[::-1][:top_k]

        similar_items = [
            self._format_candidate(self.catalog[int(p)], float(raw_sims[p]), float(blended_scores[p]))
            for p in top_pos
        ]
        return {
            "title": title,
            "genre": genre,
            "tags": merged_tags,
            "embedding": [round(float(x), 4) for x in item_vec],
            "similar_catalog_items": similar_items,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def browse_catalog(
        self,
        genre: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 60,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search and paginate the catalog across genres, titles, directors, tags, and descriptions."""
        filtered = self.catalog
        if genre:
            g_lower = genre.strip().lower()
            filtered = [it for it in filtered if it.get("genre", "").lower() == g_lower]
        if q:
            q_lower = q.strip().lower()
            filtered = [
                it
                for it in filtered
                if q_lower in it.get("title", "").lower()
                or q_lower in it.get("director", "").lower()
                or q_lower in it.get("genre", "").lower()
                or q_lower in it.get("description", "").lower()
                or any(q_lower in t.lower() for t in it.get("tags", []))
            ]
        page = filtered[offset : offset + limit]
        return {"total": len(filtered), "offset": offset, "limit": limit, "items": page}

    def get_latent_space_map(self) -> Dict[str, Any]:
        """Project catalog item embeddings into 2D PCA coordinates + export SVD basis."""
        coords, mean_vec, basis, scale = project_embeddings_2d_with_basis(self.item_embeddings)
        points = [
            {
                "id": item["id"],
                "title": item["title"],
                "genre": item["genre"],
                "year": item["year"],
                "rating": item.get("rating", 8.0),
                "x": round(float(coords[idx, 0]), 4),
                "y": round(float(coords[idx, 1]), 4),
            }
            for idx, item in enumerate(self.catalog)
        ]
        return {
            "genres": GENRES,
            "total_items": len(points),
            "items": points,
            "pca_projection": {
                "mean": [round(float(x), 5) for x in mean_vec],
                "components": [[round(float(v), 5) for v in row] for row in basis],
                "scale": round(float(scale), 5),
            },
        }

    def evaluate_metrics(
        self,
        k: int = 10,
        index_type: str = "flat",
        use_stage2: bool = False,
        diversity_lambda: float = 0.88,
    ) -> Dict[str, Any]:
        """Strict leak-free evaluation on unseen `holdout_item_ids`."""
        vec_index = self._resolve_index(index_type)

        def _ranker(user: Dict[str, Any], top_k: int) -> List[int]:
            history = set(user.get("history_item_ids", []))
            u_vec = self._encode_online_user(user)
            stage1_pool = max(22, top_k * 3) if use_stage2 else top_k
            fetch_k = min(len(self.catalog), stage1_pool + len(history) + 5)
            raw = vec_index.search(u_vec, top_k=fetch_k)
            filtered = [(self.item_id_map[iid], sc) for iid, sc in raw if iid not in history]
            if not use_stage2:
                return [m["id"] for m, _ in filtered[:top_k]]

            hist_movies = [self.item_id_map[mid] for mid in history if mid in self.item_id_map]
            reranked = self.reranker.rerank(
                user=user,
                candidates=filtered,
                item_embeddings=self.item_embeddings,
                id_to_pos=self.id_to_pos,
                top_k=top_k,
                diversity_lambda=diversity_lambda,
                history_movies=hist_movies,
            )
            return [m["id"] for m, *_ in reranked]

        metrics = evaluate_ranking_strategy(self.users, _ranker, k=k)
        metrics["index_backend"] = vec_index.backend_name
        metrics["inference_mode"] = self.inference_mode
        metrics["stage2_enabled"] = use_stage2
        return metrics
