# src/reranker.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Stage2CrossFeatureReranker:
    """
    Stage-2 Fine Re-Ranker + Maximal Marginal Relevance (MMR) Diversity Layer.

    Scores Stage-1 FAISS candidate sets using fine user-item cross features
    (cosine similarity, tag overlap Jaccard, genre match, quality prior) and
    applies iterative MMR to prevent intra-list redundancy.
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

    @staticmethod
    def _tag_overlap_score(user_tags: set, history_tags: set, item_tags: set) -> float:
        combined_user_tags = user_tags | history_tags
        if not combined_user_tags or not item_tags:
            return 0.0
        inter = len(combined_user_tags & item_tags)
        denom = len(item_tags | user_tags)
        return inter / max(1, denom)

    def rerank(
        self,
        user: Dict[str, Any],
        candidates: List[Tuple[Dict[str, Any], float]],
        item_embeddings: np.ndarray,
        id_to_pos: Dict[int, int],
        top_k: int = 6,
        diversity_lambda: float = 0.88,
        history_movies: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Compute Stage-2 cross-feature scores and select top_k items via MMR.

        Returns:
            List of `(movie_dict, stage1_cosine_score, stage2_blended_score)`.
        """
        if not candidates:
            return []

        user_tags = {t.lower().strip() for t in user.get("preferred_tags", [])}
        hist_tags = set()
        for hm in history_movies or []:
            for t in hm.get("tags", []):
                hist_tags.add(t.lower().strip())

        primary_g = user.get("primary_genre", "")
        secondary_g = user.get("secondary_genre", "")

        scored: List[Tuple[Dict[str, Any], float, float]] = []
        for movie, cos_score in candidates:
            # Map cosine similarity [-1, 1] -> [0, 1]
            cos_norm = float(np.clip((cos_score + 1.0) * 0.5, 0.0, 1.0))
            item_tags = {t.lower().strip() for t in movie.get("tags", [])}
            tag_sim = self._tag_overlap_score(user_tags, hist_tags, item_tags)

            g = movie.get("genre", "")
            genre_bonus = 1.0 if g == primary_g else (0.5 if g == secondary_g else 0.0)
            rating_norm = float(np.clip((movie.get("rating", 7.8) - 6.5) / 3.0, 0.0, 1.0))

            s2_score = (
                self.w_cosine * cos_norm
                + self.w_tag_overlap * tag_sim
                + self.w_genre_match * genre_bonus
                + self.w_quality * rating_norm
            )
            scored.append((movie, float(cos_score), float(s2_score)))

        scored.sort(key=lambda x: x[2], reverse=True)

        # Pure Stage-2 relevance ordering when lambda >= 0.999
        if diversity_lambda >= 0.999 or len(scored) <= 1:
            return scored[:top_k]

        # Maximal Marginal Relevance (MMR) greedy selection over L2 item vectors
        selected: List[Tuple[Dict[str, Any], float, float]] = []
        selected_vecs: List[np.ndarray] = []
        remaining = list(scored)

        while remaining and len(selected) < top_k:
            if not selected_vecs:
                best_idx = 0
            else:
                sel_mat = np.vstack(selected_vecs)
                best_mmr = -1e9
                best_idx = 0
                for idx, (mv, _, s2) in enumerate(remaining):
                    pos = id_to_pos.get(mv["id"])
                    if pos is None:
                        redundancy = 0.0
                    else:
                        vec = item_embeddings[pos]
                        redundancy = float(np.max(sel_mat @ vec))
                        redundancy = max(0.0, redundancy)
                    mmr_val = diversity_lambda * s2 - (1.0 - diversity_lambda) * redundancy
                    if mmr_val > best_mmr:
                        best_mmr = mmr_val
                        best_idx = idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            pos = id_to_pos.get(chosen[0]["id"])
            if pos is not None:
                selected_vecs.append(item_embeddings[pos])

        return selected
