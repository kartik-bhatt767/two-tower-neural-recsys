# api/main.py
import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.engine import RecommendationEngine

app = FastAPI(
    title="🎬 Two-Tower Neural Recommendation Microservice",
    description="FAANG-grade Deep Learning Candidate Retrieval Architecture for sub-millisecond personalized recommendations.",
    version="1.0.0"
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
rec_engine = RecommendationEngine(catalog_path=os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json"))

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def startup_event():
    rec_engine.initialize()
    print("Recommendation engine initialized and ready.")

class RecommendRequest(BaseModel):
    user_id: int = Field(default=0, ge=0, description="User index to recommend for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")
    exclude_watched: bool = Field(default=True, description="Filter out previously watched items")

@app.get("/", include_in_schema=False)
def serve_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Two-Tower RecSys API active. Visit /docs for Swagger UI."}

@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "HEALTHY",
        "service": "two-tower-neural-recsys",
        "total_users": len(rec_engine.users),
        "total_items": len(rec_engine.catalog)
    }

@app.get("/api/v1/users", tags=["Users"])
def get_users():
    return [
        {
            "user_id": u["user_id"],
            "username": u["username"],
            "primary_genre": u["primary_genre"],
            "history_count": len(u["history_item_ids"])
        }
        for u in rec_engine.users
    ]

@app.post("/api/v1/recommend", tags=["Recommendation Engine"])
def get_recommendations(req: RecommendRequest):
    return rec_engine.recommend_for_user(
        user_idx=req.user_id,
        top_k=req.top_k,
        exclude_watched=req.exclude_watched
    )

@app.get("/api/v1/metrics", tags=["Benchmarks"])
def get_evaluation_metrics():
    return rec_engine.evaluate_metrics(k=5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
