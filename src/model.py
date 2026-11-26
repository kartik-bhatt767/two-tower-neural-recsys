# src/model.py
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class UserTower(nn.Module):
        """
        Deep Neural Network mapping user ID, preferred genre, and multi-hot tag affinity
        into an L2-normalized latent embedding. Supports cold-start users via metadata-only pooling.
        """

        def __init__(
            self,
            num_users: int,
            num_genres: int,
            num_tags: int = 32,
            embed_dim: int = 32,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embed_dim)
            self.genre_embedding = nn.Embedding(num_genres, embed_dim)
            self.tag_proj = nn.Linear(num_tags, embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(
            self,
            user_idx: torch.Tensor,
            preferred_genre_idx: torch.Tensor,
            user_tag_vec: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            u_emb = self.user_embedding(user_idx)
            g_emb = self.genre_embedding(preferred_genre_idx)
            if user_tag_vec is None:
                t_emb = torch.zeros_like(u_emb)
            else:
                t_emb = self.tag_proj(user_tag_vec)
            combined = torch.cat([u_emb, g_emb, t_emb], dim=-1)
            latent = self.fc(combined)
            return F.normalize(latent, p=2, dim=-1)

        def encode_cold_user(
            self, preferred_genre_idx: torch.Tensor, user_tag_vec: torch.Tensor
        ) -> torch.Tensor:
            """Generate user embedding purely from genre + tag metadata when user has zero history."""
            mean_u_emb = self.user_embedding.weight.mean(dim=0, keepdim=True).expand(
                preferred_genre_idx.size(0), -1
            )
            g_emb = self.genre_embedding(preferred_genre_idx)
            t_emb = self.tag_proj(user_tag_vec)
            combined = torch.cat([mean_u_emb, g_emb, t_emb], dim=-1)
            latent = self.fc(combined)
            return F.normalize(latent, p=2, dim=-1)

    class ItemTower(nn.Module):
        """
        Deep Neural Network mapping item ID, genre, and content metadata tags into an
        L2-normalized latent embedding. Supports zero-click cold-start items.
        """

        def __init__(
            self,
            num_items: int,
            num_genres: int,
            num_tags: int = 32,
            embed_dim: int = 32,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.item_embedding = nn.Embedding(num_items, embed_dim)
            self.genre_embedding = nn.Embedding(num_genres, embed_dim)
            self.tag_proj = nn.Linear(num_tags, embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(
            self,
            item_idx: torch.Tensor,
            genre_idx: torch.Tensor,
            item_tag_vec: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            i_emb = self.item_embedding(item_idx)
            g_emb = self.genre_embedding(genre_idx)
            if item_tag_vec is None:
                t_emb = torch.zeros_like(i_emb)
            else:
                t_emb = self.tag_proj(item_tag_vec)
            combined = torch.cat([i_emb, g_emb, t_emb], dim=-1)
            latent = self.fc(combined)
            return F.normalize(latent, p=2, dim=-1)

        def encode_cold_item(
            self, genre_idx: torch.Tensor, item_tag_vec: torch.Tensor
        ) -> torch.Tensor:
            """Encode unseen cold-start items purely from genre + content tags without click IDs."""
            mean_i_emb = self.item_embedding.weight.mean(dim=0, keepdim=True).expand(
                genre_idx.size(0), -1
            )
            g_emb = self.genre_embedding(genre_idx)
            t_emb = self.tag_proj(item_tag_vec)
            combined = torch.cat([mean_i_emb, g_emb, t_emb], dim=-1)
            latent = self.fc(combined)
            return F.normalize(latent, p=2, dim=-1)

    class TwoTowerRecSys(nn.Module):
        """
        Two-Tower Recommendation System with Temperature-Scaled In-Batch Negative Contrastive Loss.
        """

        def __init__(
            self,
            num_users: int,
            num_items: int,
            num_genres: int,
            num_tags: int = 32,
            embed_dim: int = 32,
            temperature: float = 0.07,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_tags = num_tags
            self.temperature = temperature
            self.user_tower = UserTower(num_users, num_genres, num_tags=num_tags, embed_dim=embed_dim)
            self.item_tower = ItemTower(num_items, num_genres, num_tags=num_tags, embed_dim=embed_dim)

        def forward(
            self,
            user_ids: torch.Tensor,
            user_genres: torch.Tensor,
            item_ids: torch.Tensor,
            item_genres: torch.Tensor,
            user_tags: Optional[torch.Tensor] = None,
            item_tags: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            u_emb = self.user_tower(user_ids, user_genres, user_tags)
            v_emb = self.item_tower(item_ids, item_genres, item_tags)
            return torch.matmul(u_emb, v_emb.T) / self.temperature

        def compute_loss(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
            batch_size = similarity_matrix.size(0)
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            # Symmetric contrastive loss (user->item + item->user) stabilizes tower alignment
            loss_u2i = F.cross_entropy(similarity_matrix, labels)
            loss_i2u = F.cross_entropy(similarity_matrix.T, labels)
            return 0.5 * (loss_u2i + loss_i2u)


class TwoTowerNumPyEngine:
    """
    Pure-NumPy metadata-aware dual-tower engine combining collaborative alignment
    with genre & tag projections for fallback inference and cold-start generalization.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(42)
        self.user_embeddings = rng.standard_normal((num_users, embed_dim)).astype(np.float32)
        self.item_embeddings = rng.standard_normal((num_items, embed_dim)).astype(np.float32)
        self._normalize()

    def _normalize(self) -> None:
        u_norm = np.linalg.norm(self.user_embeddings, axis=1, keepdims=True)
        self.user_embeddings /= np.maximum(u_norm, 1e-8)
        i_norm = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        self.item_embeddings /= np.maximum(i_norm, 1e-8)

    def inject_metadata_priors(
        self,
        user_genre_indices: List[int],
        item_genre_indices: List[int],
        num_genres: int,
        user_tag_matrix: Optional[np.ndarray] = None,
        item_tag_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Align initial embeddings with deterministic genre/tag subspaces for cold-start generalization."""
        rng = np.random.default_rng(7)
        genre_basis = rng.standard_normal((num_genres, self.embed_dim)).astype(np.float32)
        genre_basis /= np.maximum(np.linalg.norm(genre_basis, axis=1, keepdims=True), 1e-8)

        for u_idx, g_idx in enumerate(user_genre_indices):
            self.user_embeddings[u_idx] = 0.25 * self.user_embeddings[u_idx] + 0.75 * genre_basis[g_idx]
        for i_idx, g_idx in enumerate(item_genre_indices):
            self.item_embeddings[i_idx] = 0.25 * self.item_embeddings[i_idx] + 0.75 * genre_basis[g_idx]

        if user_tag_matrix is not None and item_tag_matrix is not None:
            tag_proj = rng.standard_normal((item_tag_matrix.shape[1], self.embed_dim)).astype(np.float32)
            tag_proj /= np.maximum(np.linalg.norm(tag_proj, axis=1, keepdims=True), 1e-8)
            self.user_embeddings += 0.35 * (user_tag_matrix @ tag_proj)
            self.item_embeddings += 0.35 * (item_tag_matrix @ tag_proj)

        self._normalize()

    def fit_interactions(self, interactions: List[Tuple[int, int]], epochs: int = 20, lr: float = 0.08) -> None:
        """Pairwise alignment with contrastive push on sampled negatives."""
        rng = np.random.default_rng(123)
        num_items = len(self.item_embeddings)
        if num_items == 0 or len(interactions) == 0:
            return

        for _ in range(epochs):
            for u_idx, i_idx in interactions:
                u_vec = self.user_embeddings[u_idx]
                i_vec = self.item_embeddings[i_idx]
                neg_idx = int(rng.integers(0, num_items))
                if neg_idx == i_idx and num_items > 1:
                    neg_idx = (i_idx + 1) % num_items

                pos_grad = (i_vec - u_vec) * lr
                neg_grad = (self.item_embeddings[neg_idx] - u_vec) * (lr * 0.25)

                self.user_embeddings[u_idx] = u_vec + pos_grad - neg_grad
                self.item_embeddings[i_idx] = i_vec + (u_vec - i_vec) * lr
            self._normalize()

    def get_user_vector(self, user_idx: int) -> np.ndarray:
        return self.user_embeddings[user_idx]

    def get_all_item_vectors(self) -> np.ndarray:
        return self.item_embeddings

    def recommend(self, user_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        u_vec = self.get_user_vector(user_idx)
        scores = np.dot(self.item_embeddings, u_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
