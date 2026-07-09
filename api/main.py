from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.ingest import router as ingest_router
from api.routers.query import router as query_router
from core.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedding + reranker models at startup so their ~8s one-time load
    # is paid here, not inside the first /answer request. See rag.embeddings.
    from rag import embeddings

    embeddings.get_bi_encoder()
    embeddings.get_cross_encoder()
    yield
    # Flush buffered Langfuse traces on shutdown so nothing is lost when uvicorn
    # --reload restarts the worker. No-op / harmless when tracing is disabled.
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


app = FastAPI(title="Basic RAG Pipeline API", version="0.1.0", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
