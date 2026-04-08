# Thalos Prime

**Autonomous Epistemic Compiler with Synthetic Biological Intelligence (SBI)**

Thalos Prime is a production-ready, fully autonomous knowledge-processing system that
ingests raw claims from heterogeneous sources, verifies and reconciles them, stores a
versioned knowledge graph, and exposes a REST query interface — all without depending on
traditional AI models.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Thalos Prime Core                             │
│                                                                        │
│  ┌───────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │   Ingestion   │──▶│  Verification    │──▶│  Contradiction       │  │
│  │   Pipeline    │   │  Engine          │   │  Resolver            │  │
│  └───────────────┘   └──────────────────┘   └──────────┬───────────┘  │
│                                                         │              │
│  ┌─────────────────────────────────────────────────────▼───────────┐  │
│  │               Versioned Knowledge Repository                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│           ▲                           │                                │
│  ┌────────┴──────────┐   ┌────────────▼────────────┐                  │
│  │  SBI Layer        │   │  Insight Engine          │                  │
│  │  OrganoidNetwork  │   │  (gaps / low-confidence) │                  │
│  │  MicrocircuitCtrl │   └─────────────────────────┘                  │
│  └───────────────────┘                                                 │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Autonomy Layer                                                │    │
│  │  NodeRegistry · NodeSynchronizer · SelfOptimiser              │    │
│  │  AES-256-GCM + RSA-2048 encrypted inter-node communication    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Interface Layer (FastAPI)                                     │    │
│  │  POST /claims · GET /claims/{id} · GET /insights · GET /health│    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

### Layers

| Layer | Module | Responsibility |
|---|---|---|
| **Knowledge** | `thalos_prime/knowledge/` | Ingestion, verification, contradiction resolution, insight generation |
| **Storage** | `thalos_prime/storage/` | Versioned knowledge repository (JSON-persistent) |
| **SBI** | `thalos_prime/sbi/` | Organoid network simulation + microcircuit signal routing |
| **Autonomy** | `thalos_prime/autonomy/` | Distributed nodes, E2E encryption, self-optimisation |
| **Interface** | `thalos_prime/interface/` | FastAPI REST API + system monitor |
| **Core** | `thalos_prime/core.py` | `ThalosCore` orchestrator wiring all layers |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run the API server

```python
import uvicorn
from thalos_prime.core import ThalosCore
from thalos_prime.interface.api import create_app

core = ThalosCore(persist_path="knowledge.json")
app = create_app(core)
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### 3. Ingest a claim

```bash
curl -X POST http://localhost:8080/claims \
  -H "Content-Type: application/json" \
  -d '{"content": "water is H2O", "reliability": 0.95, "source_uri": "pubchem://1"}'
```

### 4. Query knowledge

```bash
curl http://localhost:8080/claims?query=water
```

### 5. Get insights

```bash
curl -X POST http://localhost:8080/insights/run
curl http://localhost:8080/insights
```

---

## Programmatic Usage

```python
from thalos_prime.core import ThalosCore

core = ThalosCore()

# Ingest a single claim
ko = core.ingest("neurons produce action potentials", reliability=0.9)
print(ko.canonical_claim.status)      # ClaimStatus.VERIFIED
print(ko.canonical_claim.confidence)  # e.g. 0.78

# Batch ingest with automatic contradiction detection
records = [
    {"content": "water is H2O",   "subject": "water", "predicate": "is", "obj": "H2O",
     "source": {"reliability": 0.95}},
    {"content": "water is H3O+",  "subject": "water", "predicate": "is", "obj": "H3O+",
     "source": {"reliability": 0.4}},
]
objects = core.ingest_batch(records)

# Run insight analysis
insights = core.run_insight_cycle()
for i in insights:
    print(i.insight_type, "-", i.description)

# Health report
print(core.health_report())

# Run one optimisation cycle (grows organoid clusters if needed)
actions = core.run_optimisation_cycle()
```

---

## Core Objectives

1. **Claim ingestion** – heterogeneous sources (text, database, experimental, sensor).
2. **Verification & provenance** – confidence scoring with full source lineage.
3. **Contradiction handling** – detection + resolution (highest-confidence / source-weighted / retain-all).
4. **Knowledge versioning** – immutable version snapshots on every update.
5. **Autonomous growth** – organoid network expands when utilisation thresholds are exceeded.
6. **Distributed operation** – encrypted node synchronisation via AES-256-GCM.
7. **Query & insight generation** – REST API + proactive gap / low-confidence detection.

---

## SBI (Synthetic Biological Intelligence) Layer

The SBI layer simulates a network of neural organoid clusters:

- **`OrganoidCluster`** – a processing unit with capacity, health, and load tracking.
- **`OrganoidNetwork`** – routes tasks to the least-loaded cluster; triggers growth when
  mean utilisation exceeds 80 %.
- **`MicrocircuitController`** – handles signal routing, analogue-to-digital encoding,
  and long-term memory via a `SynapticBuffer`.

In a physical deployment, the `OrganoidNetwork.evaluate()` method would be replaced by
actual electrophysiology read-out from lab organoid hardware.

---

## Security

- **AES-256-GCM** symmetric encryption for bulk inter-node knowledge transfers.
- **RSA-2048 OAEP** asymmetric encryption for key exchange and node identity.
- Every sync message carries additional authenticated data (AAD) tied to the sender's
  node ID, preventing replay attacks.

---

## Testing

```bash
pytest tests/ -v
```

113 tests across four test modules covering all layers.

---

## Project Layout

```
thalos_prime/
├── __init__.py
├── core.py                    # ThalosCore orchestrator
├── models/
│   └── __init__.py            # Claim, KnowledgeObject, SourceRecord, NodeInfo, …
├── knowledge/
│   ├── ingestion.py           # IngestionPipeline + ClaimNormalizer
│   ├── verification.py        # VerificationEngine
│   ├── contradiction.py       # ContradictionResolver
│   └── insight.py             # InsightEngine
├── storage/
│   └── repository.py          # KnowledgeRepository (versioned, JSON-persistent)
├── sbi/
│   ├── organoid.py            # OrganoidCluster + OrganoidNetwork
│   └── microcircuit.py        # MicrocircuitController + SynapticBuffer
├── autonomy/
│   ├── encryption.py          # SymmetricEncryption + AsymmetricEncryption
│   ├── node.py                # NodeRegistry + NodeSynchronizer
│   └── optimizer.py           # SelfOptimiser
└── interface/
    ├── api.py                 # FastAPI application factory
    └── monitor.py             # SystemMonitor
tests/
├── test_knowledge.py          # 45 tests – ingestion, verification, contradiction, repo, insights
├── test_sbi.py                # 27 tests – organoid lifecycle, network, microcircuit
├── test_autonomy.py           # 25 tests – encryption, nodes, sync, optimiser
└── test_core_and_interface.py # 16 tests – full pipeline + FastAPI endpoints
```