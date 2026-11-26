# api/main.py
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from src.data_loader import GENRES
from src.engine import RecommendationEngine

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

rec_engine = RecommendationEngine(catalog_path=CATALOG_PATH, embed_dim=32, num_users=150)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not rec_engine.catalog:
        rec_engine.initialize()
    yield


app = FastAPI(
    title="Two-Tower Neural Recommendation Microservice",
    description=(
        "High-Throughput Two-Stage Deep Learning Candidate Retrieval & Cross-Feature Re-Ranking "
        "backed by FAISS Vector Indexing for sub-millisecond personalized recommendations."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

rec_engine.initialize()

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecommendRequest(BaseModel):
    user_id: int = Field(default=0, ge=0, description="User index to recommend for")
    top_k: int = Field(default=6, ge=1, le=30, description="Number of candidates to return")
    exclude_watched: bool = Field(default=True, description="Filter out previously watched items")
    index_type: str = Field(default="hnsw", description="Vector index backend: hnsw, flat, or ivf")
    use_stage2: bool = Field(default=True, description="Enable Stage-2 Cross-Feature + MMR re-ranking")
    diversity_lambda: float = Field(default=0.88, ge=0.0, le=1.0, description="MMR relevance vs diversity tradeoff")


class ColdStartUserRequest(BaseModel):
    preferred_genre: str = Field(default="Sci-Fi", description=f"Genre preference from {GENRES}")
    tags: List[str] = Field(default_factory=lambda: ["ai", "space", "time"], description="Metadata interest tags")
    top_k: int = Field(default=6, ge=1, le=20)
    index_type: str = Field(default="flat")


class ColdStartItemRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the unseen cold-start item")
    genre: str = Field(default="Sci-Fi", description=f"Primary genre from {GENRES}")
    tags: List[str] = Field(default_factory=lambda: ["cyberpunk", "ai", "simulation"])
    top_k: int = Field(default=6, ge=1, le=20)


@app.get("/", include_in_schema=False)
def serve_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Two-Tower RecSys API active. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Monitoring"])
def health() -> Dict[str, Any]:
    return {
        "status": "HEALTHY",
        "service": "two-tower-neural-recsys",
        "inference_mode": rec_engine.inference_mode,
        "total_users": len(rec_engine.users),
        "total_items": len(rec_engine.catalog),
        "vector_indexes": {k: v.info() for k, v in rec_engine.indexes.items()},
    }


@app.get("/api/v1/users", tags=["Users"])
def get_users() -> List[Dict[str, Any]]:
    return [
        {
            "user_id": u["user_id"],
            "username": u["username"],
            "primary_genre": u["primary_genre"],
            "preferred_tags": u.get("preferred_tags", []),
            "history_count": len(u["history_item_ids"]),
            "holdout_count": len(u.get("holdout_item_ids", [])),
        }
        for u in rec_engine.users
    ]


@app.post("/api/v1/recommend", tags=["Recommendation Engine"])
def get_recommendations(req: RecommendRequest) -> Dict[str, Any]:
    try:
        return rec_engine.recommend_for_user(
            user_idx=req.user_id,
            top_k=req.top_k,
            exclude_watched=req.exclude_watched,
            index_type=req.index_type,
            use_stage2=req.use_stage2,
            diversity_lambda=req.diversity_lambda,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/similar-items/{item_id}", tags=["Recommendation Engine"])
def get_similar_items(item_id: int, top_k: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    try:
        return rec_engine.recommend_similar_items(item_id=item_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/catalog", tags=["Catalog"])
def browse_catalog(
    genre: Optional[str] = Query(default=None, description="Filter catalog by genre"),
    q: Optional[str] = Query(default=None, description="Search keyword across title, director, or tags"),
    limit: int = Query(default=60, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    return rec_engine.browse_catalog(genre=genre, q=q, limit=limit, offset=offset)


@app.post("/api/v1/cold-start/user", tags=["Cold-Start Inference"])
def cold_start_user_recommend(req: ColdStartUserRequest) -> Dict[str, Any]:
    try:
        return rec_engine.recommend_cold_start(
            preferred_genre=req.preferred_genre,
            tags=req.tags,
            top_k=req.top_k,
            index_type=req.index_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/cold-start/item", tags=["Cold-Start Inference"])
def cold_start_item_similarity(req: ColdStartItemRequest) -> Dict[str, Any]:
    return rec_engine.match_cold_start_item(
        title=req.title,
        genre=req.genre,
        tags=req.tags,
        top_k=req.top_k,
    )


@app.get("/api/v1/latent-map", tags=["Visualization"])
def get_latent_map() -> Dict[str, Any]:
    return rec_engine.get_latent_space_map()


@app.get("/api/v1/metrics", tags=["Benchmarks"])
def get_evaluation_metrics(
    k: int = Query(default=10, ge=1, le=30),
    index_type: str = Query(default="flat"),
    use_stage2: bool = Query(default=True),
) -> Dict[str, Any]:
    try:
        return rec_engine.evaluate_metrics(k=k, index_type=index_type, use_stage2=use_stage2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
