# src/model.py
import math
import numpy as np
from typing import Dict, List, Any, Tuple

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
        Deep Neural Network mapping user ID and genre history to a normalized latent embedding.
        """
        def __init__(self, num_users: int, num_genres: int, embed_dim: int = 32, hidden_dim: int = 64):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embed_dim)
            self.genre_embedding = nn.Embedding(num_genres, embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim)
            )

        def forward(self, user_idx: torch.Tensor, preferred_genre_idx: torch.Tensor) -> torch.Tensor:
            u_emb = self.user_embedding(user_idx)
            g_emb = self.genre_embedding(preferred_genre_idx)
            combined = torch.cat([u_emb, g_emb], dim=-1)
            latent = self.fc(combined)
            # L2 normalization for Cosine Dot-Product
            return F.normalize(latent, p=2, dim=-1)

    class ItemTower(nn.Module):
        """
        Deep Neural Network mapping item ID, genre, and metadata to a normalized latent embedding.
        """
        def __init__(self, num_items: int, num_genres: int, embed_dim: int = 32, hidden_dim: int = 64):
            super().__init__()
            self.item_embedding = nn.Embedding(num_items, embed_dim)
            self.genre_embedding = nn.Embedding(num_genres, embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim)
            )

        def forward(self, item_idx: torch.Tensor, genre_idx: torch.Tensor) -> torch.Tensor:
            i_emb = self.item_embedding(item_idx)
            g_emb = self.genre_embedding(genre_idx)
            combined = torch.cat([i_emb, g_emb], dim=-1)
            latent = self.fc(combined)
            return F.normalize(latent, p=2, dim=-1)

    class TwoTowerRecSys(nn.Module):
        """
        Two-Tower Recommendation System with In-Batch Negative Cross-Entropy Loss.
        Exact architecture behind YouTube & Netflix candidate retrieval.
        """
        def __init__(self, num_users: int, num_items: int, num_genres: int, embed_dim: int = 32, temperature: float = 0.07):
            super().__init__()
            self.user_tower = UserTower(num_users, num_genres, embed_dim=embed_dim)
            self.item_tower = ItemTower(num_items, num_genres, embed_dim=embed_dim)
            self.temperature = temperature

        def forward(self, user_ids, user_genres, item_ids, item_genres):
            u_emb = self.user_tower(user_ids, user_genres)
            v_emb = self.item_tower(item_ids, item_genres)
            # Similarity matrix between all users and items in batch
            similarity = torch.matmul(u_emb, v_emb.T) / self.temperature
            return similarity

        def compute_loss(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
            # Positive pairs are on the diagonal (i, i)
            batch_size = similarity_matrix.size(0)
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            loss = F.cross_entropy(similarity_matrix, labels)
            return loss

class TwoTowerNumPyEngine:
    """
    High-performance pure-NumPy inference engine for environments without GPU/heavy PyTorch,
    implementing matrix factorization and normalized dual-tower projections.
    """
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        self.embed_dim = embed_dim
        np.random.seed(42)
        # Random orthonormal vectors
        self.user_embeddings = np.random.randn(num_users, embed_dim)
        self.item_embeddings = np.random.randn(num_items, embed_dim)
        self._normalize()

    def _normalize(self):
        u_norm = np.linalg.norm(self.user_embeddings, axis=1, keepdims=True)
        self.user_embeddings /= np.maximum(u_norm, 1e-8)
        i_norm = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        self.item_embeddings /= np.maximum(i_norm, 1e-8)

    def fit_interactions(self, interactions: List[Tuple[int, int]], epochs: int = 15, lr: float = 0.05):
        """Simulates Stochastic Gradient Descent for dual tower alignment."""
        for _ in range(epochs):
            for u_idx, i_idx in interactions:
                u_vec = self.user_embeddings[u_idx]
                i_vec = self.item_embeddings[i_idx]
                # Pull positive pair closer
                grad = (i_vec - u_vec) * lr
                self.user_embeddings[u_idx] += grad
                self.item_embeddings[i_idx] += grad
            self._normalize()

    def get_user_vector(self, user_idx: int) -> np.ndarray:
        return self.user_embeddings[user_idx]

    def get_all_item_vectors(self) -> np.ndarray:
        return self.item_embeddings

    def recommend(self, user_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        u_vec = self.get_user_vector(user_idx)
        # Cosine dot-product across all items in parallel (O(N * D))
        scores = np.dot(self.item_embeddings, u_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
