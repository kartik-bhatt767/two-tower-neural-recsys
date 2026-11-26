# src/reranker.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Stage2CrossFeatureReranker:
    """
    Vectorized Stage-2 Fine Re-Ranker + Maximal Marginal Relevance (MMR) Diversity Layer.

    Computes cross-features over Stage-1 FAISS candidate sets in vectorized NumPy
    and applies O(N) incremental MMR similarity tracking using a single precomputed
    pairwise similarity matrix.
    """

    def __init__(
        self,
        w_cosine: float = 0.62,
        w_tag_overlap: float = 0.20,
        w_genre_match: float = 0.12,
        w_quality: float = 0.06,
    ):
        self.w_cosine = w_cosine
        self.w_tag_overlap = w_tag_overlap
        self.w_genre_match = w_genre_match
        self.w_quality = w_quality

    def rerank(
        self,
        user: Dict[str, Any],
        candidates: List[Tuple[Dict[str, Any], float]],
        item_embeddings: np.ndarray,
        id_to_pos: Dict[int, int],
        top_k: int = 6,
        diversity_lambda: float = 0.88,
        history_movies: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[Dict[str, Any], float, float, int, int]]:
        """
        Compute Stage-2 cross-feature scores and select top_k items via vectorized MMR.

        Returns:
            List of `(movie_dict, stage1_cosine, stage2_score, stage1_rank, final_rank)`.
        """
        n = len(candidates)
        if n == 0:
            return []

        user_tags = {t.lower().strip() for t in user.get("preferred_tags", [])}
        hist_tags = set()
        for hm in history_movies or []:
            for t in hm.get("tags", []):
                hist_tags.add(t.lower().strip())
        combined_user_tags = user_tags | hist_tags

        primary_g = user.get("primary_genre", "")
        secondary_g = user.get("secondary_genre", "")

        # Vectorized cross-feature assembly over Stage-1 pool
        cos_scores = np.array([sc for _, sc in candidates], dtype=np.float32)
        cos_norm = np.clip((cos_scores + 1.0) * 0.5, 0.0, 1.0)

        tag_sims = np.zeros(n, dtype=np.float32)
        genre_bonuses = np.zeros(n, dtype=np.float32)
        rating_norms = np.empty(n, dtype=np.float32)
        cand_positions = np.empty(n, dtype=np.int32)

        for idx, (mv, _) in enumerate(candidates):
            itags = {t.lower().strip() for t in mv.get("tags", [])}
            if combined_user_tags and itags:
                denom = max(1, len(itags | user_tags))
                tag_sims[idx] = len(combined_user_tags & itags) / denom

            g = mv.get("genre", "")
            if g == primary_g:
                genre_bonuses[idx] = 1.0
            elif g == secondary_g:
                genre_bonuses[idx] = 0.5

            rating_norms[idx] = min(1.0, max(0.0, (mv.get("rating", 7.8) - 6.5) / 3.0))
            cand_positions[idx] = id_to_pos.get(mv["id"], 0)

        s2_scores = (
            self.w_cosine * cos_norm
            + self.w_tag_overlap * tag_sims
            + self.w_genre_match * genre_bonuses
            + self.w_quality * rating_norms
        )

        # Stage-1 ordering preserved via index+1
        if diversity_lambda >= 0.999 or n <= 1:
            order = np.argsort(-s2_scores, kind="stable")[:top_k]
            return [
                (
                    candidates[i][0],
                    float(cos_scores[i]),
                    float(s2_scores[i]),
                    int(i + 1),
                    int(final_r),
                )
                for final_r, i in enumerate(order, start=1)
            ]

        # Precompute candidate-to-candidate dot products once in BLAS
        cand_vecs = item_embeddings[cand_positions]
        sim_matrix = cand_vecs @ cand_vecs.T

        first_idx = int(np.argmax(s2_scores))
        chosen_indices: List[int] = [first_idx]
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[first_idx] = True
        max_sim_to_selected = np.maximum(0.0, sim_matrix[first_idx].copy())

        limit = min(top_k, n)
        while len(chosen_indices) < limit:
            mmr_scores = diversity_lambda * s2_scores - (1.0 - diversity_lambda) * max_sim_to_selected
            mmr_scores[selected_mask] = -1e9
            best_idx = int(np.argmax(mmr_scores))
            chosen_indices.append(best_idx)
            selected_mask[best_idx] = True
            np.maximum(max_sim_to_selected, np.maximum(0.0, sim_matrix[best_idx]), out=max_sim_to_selected)

        return [
            (
                candidates[i][0],
                float(cos_scores[i]),
                float(s2_scores[i]),
                int(i + 1),
                int(final_r),
            )
            for final_r, i in enumerate(chosen_indices, start=1)
        ]
