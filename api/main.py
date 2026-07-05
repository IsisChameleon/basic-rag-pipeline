from fastapi import FastAPI

from api.routers.ingest import router as ingest_router
from api.routers.query import router as query_router

app = FastAPI(title="Basic RAG Pipeline API", version="0.1.0")

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
