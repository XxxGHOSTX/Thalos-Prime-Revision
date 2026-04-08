"""
Query and Insight API
======================

FastAPI-based HTTP interface for Thalos Prime.

Endpoints:
  POST /claims          – ingest a new claim
  GET  /claims/{id}     – retrieve a knowledge object
  GET  /claims/search   – keyword search
  GET  /insights        – retrieve top insights
  POST /insights/run    – trigger an insight analysis cycle
  GET  /health          – node and organoid health report
  GET  /nodes           – list registered peer nodes
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class IngestRequest(BaseModel):  # type: ignore[misc]
        content: str
        source_uri: str = "api://manual"
        source_type: str = "text"
        reliability: float = 0.7
        tags: list[str] = []
        metadata: dict[str, Any] = {}

    class SearchRequest(BaseModel):  # type: ignore[misc]
        query: str
        limit: int = 20


def create_app(thalos_core: Any) -> Any:
    """
    Build and return the FastAPI application wired to a ThalosCore instance.

    Args:
        thalos_core: An initialised :class:`~thalos_prime.core.ThalosCore`.
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed. Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Thalos Prime API",
        description="Autonomous Epistemic Compiler – knowledge query interface",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # Claim endpoints
    # ------------------------------------------------------------------

    @app.post("/claims", status_code=201)
    async def ingest_claim(req: IngestRequest) -> dict[str, Any]:  # type: ignore[name-defined]
        """Ingest a new claim into Thalos Prime."""
        ko = thalos_core.ingest(
            content=req.content,
            source_uri=req.source_uri,
            source_type=req.source_type,
            reliability=req.reliability,
            tags=req.tags,
            metadata=req.metadata,
        )
        return {
            "object_id": ko.object_id,
            "claim_id": ko.canonical_claim.claim_id,
            "status": ko.canonical_claim.status.value,
            "confidence": ko.canonical_claim.confidence,
        }

    @app.get("/claims/{object_id}")
    async def get_claim(object_id: str) -> dict[str, Any]:
        """Retrieve a knowledge object by ID."""
        ko = thalos_core.repository.get(object_id)
        if ko is None:
            raise HTTPException(status_code=404, detail="Knowledge object not found")
        return {
            "object_id": ko.object_id,
            "content": ko.canonical_claim.content,
            "subject": ko.canonical_claim.subject,
            "predicate": ko.canonical_claim.predicate,
            "obj": ko.canonical_claim.obj,
            "confidence": ko.canonical_claim.confidence,
            "status": ko.canonical_claim.status.value,
            "versions": len(ko.versions),
            "created_at": ko.created_at.isoformat(),
            "updated_at": ko.updated_at.isoformat(),
        }

    @app.get("/claims")
    async def search_claims(query: str = "", limit: int = 20) -> dict[str, Any]:
        """Search knowledge objects by keyword."""
        results = thalos_core.repository.search(query, limit=limit)
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "object_id": ko.object_id,
                    "content": ko.canonical_claim.content[:120],
                    "confidence": ko.canonical_claim.confidence,
                    "status": ko.canonical_claim.status.value,
                }
                for ko in results
            ],
        }

    # ------------------------------------------------------------------
    # Insight endpoints
    # ------------------------------------------------------------------

    @app.get("/insights")
    async def get_insights(limit: int = 10) -> dict[str, Any]:
        """Return the top insights."""
        insights = thalos_core.insight_engine.top_insights(n=limit)
        return {
            "count": len(insights),
            "insights": [
                {
                    "insight_id": i.insight_id,
                    "type": i.insight_type,
                    "description": i.description,
                    "priority": i.priority,
                    "related_claims": len(i.related_claim_ids),
                }
                for i in insights
            ],
        }

    @app.post("/insights/run")
    async def run_insights() -> dict[str, Any]:
        """Trigger an insight analysis cycle."""
        new_insights = thalos_core.run_insight_cycle()
        return {"new_insights": len(new_insights)}

    # ------------------------------------------------------------------
    # Health / monitoring
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Node and organoid health report."""
        return thalos_core.health_report()

    @app.get("/nodes")
    async def list_nodes() -> dict[str, Any]:
        """List registered peer nodes."""
        nodes = thalos_core.node_registry.all_peers
        return {
            "local_node_id": thalos_core.node_registry.local_node.node_id,
            "peer_count": len(nodes),
            "peers": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "status": n.status,
                    "last_seen": n.last_seen.isoformat(),
                }
                for n in nodes
            ],
        }

    return app
