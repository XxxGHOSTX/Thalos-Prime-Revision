from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from thalos_prime.config import get_settings
from thalos_prime.storage.db import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    init_db(settings.database_url)
    logger.info("Thalos Prime API started")
    yield
    logger.info("Thalos Prime API shutdown")


app = FastAPI(title="Thalos Prime API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    url: Optional[str] = None
    file_b64: Optional[str] = None
    file_name: Optional[str] = None
    feed: Optional[str] = None


class QueryRequest(BaseModel):
    text: str
    top_k: int = 10
    lang: Optional[str] = None


class FeedRequest(BaseModel):
    url: str
    feed_type: str = "rss"


def _get_core():
    from thalos_prime.core import ThalosCore
    settings = get_settings()
    return ThalosCore(settings)


@app.get("/health")
async def health():
    from sqlalchemy import select, func
    from thalos_prime.storage.db import Claim, get_session
    settings = get_settings()
    with get_session(settings.database_url) as session:
        total = session.execute(select(func.count(Claim.id))).scalar() or 0
    return {"status": "ok", "version": "0.1.0", "claims_total": total}


@app.post("/ingest")
async def ingest(request: IngestRequest):
    core = _get_core()
    if request.url:
        result = core.ingest_url(request.url)
        return {"status": result.status, "doc_id": result.doc_id, "word_count": result.word_count}
    elif request.feed:
        results = core.ingest_feed(request.feed)
        return {"status": "ok", "results": [{"doc_id": r.doc_id, "status": r.status} for r in results]}
    elif request.file_b64:
        import tempfile
        from pathlib import Path
        file_bytes = base64.b64decode(request.file_b64)
        suffix = Path(request.file_name or "file.txt").suffix
        tmp_path = get_settings().data_dir / f"upload_{hash(request.file_b64)}{suffix}"
        tmp_path.write_bytes(file_bytes)
        try:
            result = core.ingest_file(str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return {"status": result.status, "doc_id": result.doc_id, "word_count": result.word_count}
    raise HTTPException(status_code=400, detail="Provide url, feed, or file_b64")


@app.post("/pipeline/{doc_id}")
async def run_pipeline(doc_id: str):
    core = _get_core()
    result = core.run_pipeline(doc_id)
    return result


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    core = _get_core()
    result = core.query(request.text, top_k=request.top_k)
    return result


@app.get("/claims/{coordinate}")
async def get_claim(coordinate: str):
    core = _get_core()
    claim = core.get_claim(coordinate)
    if claim is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


@app.get("/claims")
async def list_claims(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
):
    from sqlalchemy import select
    from thalos_prime.storage.db import Claim, get_session
    settings = get_settings()
    with get_session(settings.database_url) as session:
        q = select(Claim)
        if status:
            q = q.where(Claim.status == status)
        q = q.offset(offset).limit(limit)
        claims = session.execute(q).scalars().all()
        return [
            {
                "id": c.id,
                "coordinate": c.coordinate,
                "text": c.text[:300],
                "status": c.status,
                "language": c.language,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in claims
        ]


@app.post("/feeds")
async def register_feed(request: FeedRequest):
    from thalos_prime.ingestion.api_client import FeedClient
    settings = get_settings()
    client = FeedClient(settings)
    feed_id = client.register_feed(request.url, request.feed_type)
    return {"feed_id": feed_id, "url": request.url, "feed_type": request.feed_type}


@app.delete("/claims/{coordinate}/retract")
async def retract_claim(coordinate: str):
    core = _get_core()
    success = core.retract_claim(coordinate)
    if not success:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"status": "retracted", "coordinate": coordinate}
