# src/vector_index.py
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.cluster import KMeans


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-8)


class BaseVectorIndex:
    """Common interface for exact and approximate Maximum Inner Product Search (MIPS)."""

    backend_name: str = "base"

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.item_ids: List[int] = []
        self.vectors: Optional[np.ndarray] = None
        self.build_time_ms: float = 0.0

    def build(self, embeddings: np.ndarray, item_ids: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "num_vectors": len(self.item_ids),
            "embed_dim": self.embed_dim,
            "build_time_ms": round(self.build_time_ms, 3),
        }


class FAISSFlatIPIndex(BaseVectorIndex):
    """Exact Maximum Inner Product Search backed by FAISS IndexFlatIP (or NumPy fallback)."""

    backend_name = "faiss_flat" if FAISS_AVAILABLE else "numpy_flat"

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.index = None

    def build(self, embeddings: np.ndarray, item_ids: Optional[List[int]] = None) -> None:
        t0 = time.perf_counter()
        vecs = _normalize_matrix(embeddings)
        self.vectors = vecs
        self.item_ids = list(item_ids) if item_ids is not None else list(range(len(vecs)))

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embed_dim)
            self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.build_time_ms = (time.perf_counter() - t0) * 1000.0

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.item_ids) == 0:
            return []
        k = min(top_k, len(self.item_ids))
        q = np.ascontiguousarray(_normalize_vector(query_vector), dtype=np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q, k)
            return [
                (self.item_ids[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]

        scores = (self.vectors @ q[0]).astype(np.float32)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.item_ids[int(i)], float(scores[i])) for i in top_idx]


class FAISSHNSWIndex(BaseVectorIndex):
    """Hierarchical Navigable Small World (HNSW) graph ANN index for sub-millisecond MIPS."""

    backend_name = "faiss_hnsw" if FAISS_AVAILABLE else "numpy_flat"

    def __init__(self, embed_dim: int, m_links: int = 16, ef_construction: int = 64, ef_search: int = 32):
        super().__init__(embed_dim)
        self.m_links = m_links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None

    def build(self, embeddings: np.ndarray, item_ids: Optional[List[int]] = None) -> None:
        t0 = time.perf_counter()
        vecs = _normalize_matrix(embeddings)
        self.vectors = vecs
        self.item_ids = list(item_ids) if item_ids is not None else list(range(len(vecs)))

        if FAISS_AVAILABLE:
            self.index = faiss.IndexHNSWFlat(self.embed_dim, self.m_links, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.build_time_ms = (time.perf_counter() - t0) * 1000.0

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.item_ids) == 0:
            return []
        k = min(top_k, len(self.item_ids))
        q = np.ascontiguousarray(_normalize_vector(query_vector), dtype=np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q, k)
            return [
                (self.item_ids[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]

        scores = (self.vectors @ q[0]).astype(np.float32)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.item_ids[int(i)], float(scores[i])) for i in top_idx]


class IVFFlatIndex(BaseVectorIndex):
    """
    Inverted File (IVF-Flat) Voronoi cluster ANN index.
    Uses FAISS IndexIVFFlat when available, and Scikit-Learn/NumPy Voronoi probing as fallback.
    """

    backend_name = "faiss_ivf" if FAISS_AVAILABLE else "numpy_ivf"

    def __init__(self, embed_dim: int, nlist: int = 8, nprobe: int = 3):
        super().__init__(embed_dim)
        self.requested_nlist = nlist
        self.requested_nprobe = nprobe
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = {}

    def build(self, embeddings: np.ndarray, item_ids: Optional[List[int]] = None) -> None:
        t0 = time.perf_counter()
        vecs = _normalize_matrix(embeddings)
        self.vectors = vecs
        self.item_ids = list(item_ids) if item_ids is not None else list(range(len(vecs)))
        n_items = len(vecs)

        # Ensure nlist leaves sufficient points per Voronoi cell for reliable probing
        self.nlist = max(1, min(self.requested_nlist, max(1, n_items // 10)))
        self.nprobe = max(1, min(self.requested_nprobe, self.nlist))

        if FAISS_AVAILABLE and self.nlist >= 2 and n_items >= self.nlist:
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            ivf = faiss.IndexIVFFlat(quantizer, self.embed_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            ivf.train(np.ascontiguousarray(vecs, dtype=np.float32))
            ivf.add(np.ascontiguousarray(vecs, dtype=np.float32))
            ivf.nprobe = self.nprobe
            self.index = ivf
            self.centroids = None
        else:
            # Pure Scikit-Learn / NumPy Voronoi partitioning
            self.index = None
            if self.nlist > 1 and n_items >= self.nlist:
                km = KMeans(n_clusters=self.nlist, n_init=5, random_state=42)
                labels = km.fit_predict(vecs)
                self.centroids = _normalize_matrix(km.cluster_centers_)
                self.inverted_lists = {c: [] for c in range(self.nlist)}
                for pos, c_id in enumerate(labels):
                    self.inverted_lists[int(c_id)].append(pos)
            else:
                self.centroids = np.mean(vecs, axis=0, keepdims=True)
                self.inverted_lists = {0: list(range(n_items))}

        self.build_time_ms = (time.perf_counter() - t0) * 1000.0

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.item_ids) == 0:
            return []
        k = min(top_k, len(self.item_ids))
        q = np.ascontiguousarray(_normalize_vector(query_vector), dtype=np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q, k)
            results = [
                (self.item_ids[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]
            if len(results) < k:
                # Broaden probe window if initial Voronoi cells hold fewer than k candidates
                self.index.nprobe = self.nlist
                scores, indices = self.index.search(q, k)
                self.index.nprobe = self.nprobe
                results = [
                    (self.item_ids[int(idx)], float(score))
                    for idx, score in zip(indices[0], scores[0])
                    if idx >= 0
                ]
            return results[:k]

        if self.centroids is None:
            scores = (self.vectors @ q[0]).astype(np.float32)
            top_idx = np.argsort(scores)[::-1][:k]
            return [(self.item_ids[int(i)], float(scores[i])) for i in top_idx]

        # Probe top-nprobe Voronoi clusters (NumPy/Scikit-Learn path)
        centroid_scores = self.centroids @ q[0]
        probed_clusters = np.argsort(centroid_scores)[::-1][: self.nprobe]
        candidate_positions: List[int] = []
        for c_id in probed_clusters:
            candidate_positions.extend(self.inverted_lists.get(int(c_id), []))

        if len(candidate_positions) < k:
            candidate_positions = list(range(len(self.item_ids)))

        cand_vecs = self.vectors[candidate_positions]
        cand_scores = cand_vecs @ q[0]
        top_local = np.argsort(cand_scores)[::-1][:k]
        return [
            (self.item_ids[candidate_positions[int(pos)]], float(cand_scores[pos]))
            for pos in top_local
        ]


def create_vector_index(index_type: str, embed_dim: int) -> BaseVectorIndex:
    """Factory creating a configured MIPS vector index by type."""
    normalized = index_type.lower().strip()
    if normalized in ("flat", "faiss_flat", "exact"):
        return FAISSFlatIPIndex(embed_dim=embed_dim)
    if normalized in ("hnsw", "faiss_hnsw"):
        return FAISSHNSWIndex(embed_dim=embed_dim)
    if normalized in ("ivf", "faiss_ivf", "ivf_flat"):
        return IVFFlatIndex(embed_dim=embed_dim)
    raise ValueError(f"Unsupported vector index type: {index_type}")


def project_embeddings_2d(embeddings: np.ndarray) -> np.ndarray:
    """Compute centered, [-1, 1] scaled 2D PCA coordinates using SVD."""
    vecs = np.asarray(embeddings, dtype=np.float32)
    if len(vecs) < 2:
        return np.zeros((len(vecs), 2), dtype=np.float32)
    centered = vecs - np.mean(vecs, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    max_abs = np.max(np.abs(coords))
    if max_abs > 1e-8:
        coords = (coords / max_abs) * 0.88
    return coords
