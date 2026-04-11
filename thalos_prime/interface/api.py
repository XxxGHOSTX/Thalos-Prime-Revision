"""
Query and Insight API
======================

FastAPI-based HTTP interface for Thalos Prime.

Endpoints:
  POST   /claims              – ingest a new claim
  GET    /claims              – keyword search across knowledge objects
  GET    /claims/{id}         – retrieve a knowledge object
  PUT    /claims/{id}         – re-ingest / update a knowledge object
  DELETE /claims/{id}         – delete a knowledge object
  GET    /claims/{id}/versions – full version history for a knowledge object
  GET    /insights            – retrieve top insights
  POST   /insights/run        – trigger an insight analysis cycle
  GET    /health              – node and organoid health report
  GET    /optimiser           – self-optimiser status and action history
  GET    /nodes               – list registered peer nodes
  POST   /nodes               – register a peer node
  POST   /nodes/{id}/heartbeat – send a heartbeat for a peer node
  POST   /nodes/prune         – mark stale nodes as offline
  GET    /sync                – prepare an outgoing sync payload (JSON)
  POST   /sync                – receive an encrypted sync payload from a peer
  GET    /monitor             – full system health snapshot
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

    class UpdateRequest(BaseModel):  # type: ignore[misc]
        content: str
        source_uri: str = "api://update"
        reliability: float = 0.7
        tags: list[str] = []

    class RegisterNodeRequest(BaseModel):  # type: ignore[misc]
        host: str
        port: int = 8080
        public_key_pem: str = ""
        node_id: str = ""

    class SyncReceiveRequest(BaseModel):  # type: ignore[misc]
        """Wrapper for an encrypted sync payload delivered as base-64."""
        payload_b64: str


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

    import base64

    from thalos_prime.models import NodeInfo

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

    @app.get("/claims/{object_id}/versions")
    async def get_claim_versions(object_id: str) -> dict[str, Any]:
        """Return the full version history for a knowledge object."""
        ko = thalos_core.repository.get(object_id)
        if ko is None:
            raise HTTPException(status_code=404, detail="Knowledge object not found")
        return {
            "object_id": ko.object_id,
            "version_count": len(ko.versions),
            "versions": [
                {
                    "version_number": v.version_number,
                    "version_id": v.version_id,
                    "reason": v.reason,
                    "created_at": v.created_at.isoformat(),
                    "content": v.claim_snapshot.get("content", ""),
                    "confidence": v.claim_snapshot.get("confidence"),
                    "status": v.claim_snapshot.get("status"),
                }
                for v in ko.versions
            ],
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
            "tags": ko.canonical_claim.tags,
            "versions": len(ko.versions),
            "created_at": ko.created_at.isoformat(),
            "updated_at": ko.updated_at.isoformat(),
            "sbi_cluster": ko.canonical_claim.metadata.get("sbi_cluster", ""),
        }

    @app.put("/claims/{object_id}")
    async def update_claim(object_id: str, req: UpdateRequest) -> dict[str, Any]:  # type: ignore[name-defined]
        """
        Re-ingest updated content for an existing knowledge object.

        The existing object is replaced with a freshly ingested version
        at the same ``object_id``; a new version snapshot is created.
        """
        existing = thalos_core.repository.get(object_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Knowledge object not found")

        # Ingest as a fresh claim, then graft it onto the existing object
        ko_new = thalos_core.ingest(
            content=req.content,
            source_uri=req.source_uri,
            reliability=req.reliability,
            tags=req.tags,
        )
        # Transfer content + status onto the existing object (preserves history)
        existing.canonical_claim.content = ko_new.canonical_claim.content
        existing.canonical_claim.confidence = ko_new.canonical_claim.confidence
        existing.canonical_claim.status = ko_new.canonical_claim.status
        existing.canonical_claim.tags = ko_new.canonical_claim.tags
        existing.canonical_claim.metadata.update(ko_new.canonical_claim.metadata)
        existing.canonical_claim.touch()
        # Persist the update (creates a new version)
        thalos_core.repository.put(existing, reason="api_update")
        # Remove the temporary new object
        thalos_core.repository.delete(ko_new.object_id)
        return {
            "object_id": existing.object_id,
            "status": existing.canonical_claim.status.value,
            "confidence": existing.canonical_claim.confidence,
            "versions": len(existing.versions),
        }

    @app.delete("/claims/{object_id}", status_code=204)
    async def delete_claim(object_id: str) -> None:
        """Permanently delete a knowledge object."""
        deleted = thalos_core.repository.delete(object_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Knowledge object not found")

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

    @app.get("/optimiser")
    async def optimiser_status() -> dict[str, Any]:
        """Self-optimiser status and recent action history."""
        return thalos_core.optimiser.report()

    @app.get("/monitor")
    async def monitor_snapshot() -> dict[str, Any]:
        """Capture and return a full system health snapshot."""
        return thalos_core.monitor.snapshot()

    @app.get("/monitor/history")
    async def monitor_history(limit: int = 10) -> dict[str, Any]:
        """Return recent system health snapshots."""
        snapshots = thalos_core.monitor.recent_snapshots(n=limit)
        return {"count": len(snapshots), "snapshots": snapshots}

    # ------------------------------------------------------------------
    # Node management endpoints
    # ------------------------------------------------------------------

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

    @app.post("/nodes", status_code=201)
    async def register_node(req: RegisterNodeRequest) -> dict[str, Any]:  # type: ignore[name-defined]
        """Register a peer node."""
        from thalos_prime.models import NodeInfo

        node = NodeInfo(
            host=req.host,
            port=req.port,
            public_key_pem=req.public_key_pem,
        )
        if req.node_id:
            node.node_id = req.node_id
        thalos_core.register_peer(node)
        return {
            "registered": True,
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
        }

    @app.post("/nodes/{node_id}/heartbeat")
    async def node_heartbeat(node_id: str) -> dict[str, Any]:
        """Send a liveness heartbeat for a peer node."""
        peer = thalos_core.node_registry.get_peer(node_id)
        if peer is None:
            raise HTTPException(status_code=404, detail="Peer node not found")
        thalos_core.node_registry.heartbeat(node_id)
        return {"node_id": node_id, "status": "heartbeat_received"}

    @app.post("/nodes/prune")
    async def prune_nodes() -> dict[str, Any]:
        """Mark stale nodes (no heartbeat within timeout) as offline."""
        stale = thalos_core.node_registry.prune_stale()
        return {"stale_nodes_marked_offline": len(stale), "node_ids": stale}

    # ------------------------------------------------------------------
    # Sync endpoints
    # ------------------------------------------------------------------

    @app.get("/sync")
    async def prepare_sync() -> dict[str, Any]:
        """
        Prepare a sync payload for this node.

        Returns the encrypted bytes base-64 encoded so they can be
        transmitted as JSON and POSTed to a peer's ``/sync`` endpoint.
        """
        encrypted = thalos_core.prepare_sync_payload()
        return {
            "sender_node_id": thalos_core.node_registry.local_node.node_id,
            "knowledge_objects": thalos_core.repository.count(),
            "payload_b64": base64.b64encode(encrypted).decode(),
        }

    @app.post("/sync")
    async def receive_sync(req: SyncReceiveRequest) -> dict[str, Any]:  # type: ignore[name-defined]
        """
        Receive an encrypted sync payload from a peer node.

        The payload must be the base-64 encoded bytes produced by the
        sending node's ``GET /sync`` endpoint.
        """
        try:
            encrypted = base64.b64decode(req.payload_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base-64 payload")
        merged = thalos_core.receive_sync_payload(encrypted)
        return {
            "merged_objects": merged,
            "total_objects": thalos_core.repository.count(),
        }

    return app
