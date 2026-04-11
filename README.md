# Thalos Prime Revision

**An Autonomous Epistemic Compiler** — a self-organizing knowledge pipeline that ingests text from any source, extracts factual claims, links evidence, detects contradictions, verifies knowledge, and answers questions.

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│  Ingestion  │──▶│Normalization│──▶│ Claim Extraction │
│  (web/file/ │   │ (clean/lang │   │ (spaCy NLP +     │
│   feed/rss) │   │  detect/    │   │  factual filter) │
└─────────────┘   │  translate) │   └────────┬─────────┘
                  └─────────────┘            │
                                             ▼
┌──────────────┐  ┌──────────────┐  ┌───────────────────┐
│  Autonomy    │  │ Query/Answer │  │  Evidence Linking │
│  Loop        │◀─│ (BM25+       │  │  (BM25 + semantic │
│  (APScheduler│  │  semantic +  │  │   cosine + NLI)   │
│  hourly/etc) │  │  RRF fusion) │  └────────┬──────────┘
└──────────────┘  └──────────────┘           │
                                             ▼
┌──────────────────┐  ┌─────────────┐  ┌────────────────────┐
│  Verification &  │◀─│Contradiction│◀─│ Deterministic      │
│  Promotion       │  │ Detection   │  │ Coordinates        │
│  (multi-signal   │  │ (NLI cross- │  │ (TP:<base58(sha256)│
│   scoring)       │  │  encoder)   │  │  12-char)          │
└──────────────────┘  └─────────────┘  └────────────────────┘
```

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

## CLI Usage

```bash
# Ingest a URL
thalos-prime ingest --url https://en.wikipedia.org/wiki/Python_(programming_language)

# Ingest a local file
thalos-prime ingest --file ./document.txt

# Ingest an RSS feed
thalos-prime ingest --feed https://feeds.bbci.co.uk/news/rss.xml

# Run pipeline on an already-ingested document
thalos-prime pipeline --doc-id <uuid>

# Query the knowledge base
thalos-prime query "What is the boiling point of water?"

# Start the API server
thalos-prime serve --port 8000

# Check database statistics
thalos-prime status

# Run evaluation suite
thalos-prime eval --suite all --limit 10

# Interactive shell
thalos-prime shell
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest a URL
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Python?", "top_k": 5}'

# Get claim by coordinate
curl http://localhost:8000/claims/TP:AbCdEfGhIjKl

# List claims
curl "http://localhost:8000/claims?status=verified&limit=20"

# Retract a claim
curl -X DELETE http://localhost:8000/claims/TP:AbCdEfGhIjKl/retract
```

## Configuration

Copy `.env.example` to `.env` and customize. Key settings (all prefixed with `THALOS_`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./thalos.db` | SQLAlchemy DB URL |
| `DATA_DIR` | `./thalos_data` | Data directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `USE_CLOUD_LLM` | `false` | Use OpenAI for answers |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model |

## Pipeline Stages

1. **Ingestion**: Web crawler (httpx + BS4), file loader (PDF/DOCX/TXT), RSS/JSON feeds. SHA-256 deduplication.
2. **Normalization**: HTML stripping, language detection, translation to English.
3. **Claim Extraction**: spaCy NLP factual sentence filter + deterministic coordinate assignment.
4. **Deterministic Coordinates**: `TP:<12-char-base58(sha256(lang::text))>`
5. **Evidence Linking**: BM25 + semantic cosine similarity, NLI classification.
6. **Contradiction Detection**: NLI cross-encoder with cosine similarity pre-filter.
7. **Verification & Promotion**: Multi-signal scoring with promotion gates.
8. **Query/Answer**: Hybrid BM25+semantic RRF fusion. Template or LLM synthesis.
9. **Autonomy Loop**: Background feed re-ingestion, stale re-verification, contradiction checking.

## Development

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=thalos_prime --cov-report=term-missing
```
