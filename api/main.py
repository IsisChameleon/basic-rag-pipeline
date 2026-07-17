from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.container import build_api_container
from api.routers.ingest import router as ingest_router
from api.routers.query import router as query_router
from core.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_api_container()  # cheap: nothing heavy loads here
    # Warm the embedding + reranker models now so their ~8s one-time load is
    # paid at boot (fail-fast, fast first request), not inside the first
    # /search. Runs before any request thread exists, so the lazy first load
    # in the encoders needs no locking.
    container.warm_up()
    app.state.container = container
    yield
    # Flush buffered Langfuse traces on shutdown so nothing is lost when
    # uvicorn --reload restarts the worker. No-op / harmless when tracing is
    # disabled.
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
