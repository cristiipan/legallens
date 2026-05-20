# LegalLens

A distributed agent-powered contract review platform. An LLM agent
autonomously identifies risk clauses and answers follow-up questions, with
work fanned out across a pool of workers coordinated through Redis Sentinel
and a consistent-hash ring. Built on the public
[CUAD dataset](https://www.atticusprojectai.org/cuad) (510 commercial
contracts, 13,000+ expert annotations across 41 clause categories).

> **Status:** scaffold + core modules implemented; end-to-end ingest +
> agent loop wiring in progress. See [Roadmap](#roadmap).

## Why distributed?

A single-process contract reviewer is fine until you want to ingest 500+
contracts in parallel, or run multiple concurrent reviews without each one
blocking on the same Cohere/Pinecone connections. The architecture below
isolates that work in a worker pool:

- **Consistent hashing** routes every task for a given `contract_id` to the
  same worker — that worker can cache the parsed contract, reuse open DB
  connections to the right Postgres partition, and avoid double-embedding.
- **Redis Sentinel** keeps the orchestration plane (queues, heartbeats,
  pub/sub) available across a Redis master failure.
- **Hash-partitioned Postgres** (8 partitions of `contracts` and `clauses`,
  partitioned on `HASH(contract_id)`) makes parallel writes contention-free.
- **Server-Sent Events** stream the agent's reasoning to the client in real
  time, so a 10–30s review feels responsive instead of opaque.

## Architecture

```
                ┌──────────────┐      SSE stream      ┌──────────────────────┐
                │   Client     │ ◄──────────────────  │    FastAPI / SSE     │
                └──────────────┘                      └──────────┬───────────┘
                                                                 │ publish
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │   Redis (Sentinel)   │
                                                      │  queues + heartbeats │
                                                      │  + pub/sub events    │
                                                      └─┬────────────┬───────┘
                                                        │            │
                                       ┌────────────────┤            ├────────────────┐
                                       │  consistent    │            │ consistent     │
                                       │  hash ring     ▼            ▼ hash ring      │
                                       │     ┌────────────────┐  ┌────────────────┐   │
                                       │     │  Worker A      │  │  Worker B      │   │
                                       │     │  (agent loop / │  │  (agent loop / │   │
                                       │     │   ingest)      │  │   ingest)      │   │
                                       │     └─────┬──────────┘  └─────┬──────────┘   │
                                       │           │ tool calls        │ tool calls   │
                                       │           ▼                   ▼              │
                                       │   ┌────────────────────────────────────┐     │
                                       │   │   Cohere Command R   (tool use)    │     │
                                       │   └────────────────────────────────────┘     │
                                       │           │                                  │
                                       │           ├────────────┬─────────────────────┤
                                       │           ▼            ▼                     │
                                       │   ┌──────────────┐  ┌──────────────────┐     │
                                       │   │   Pinecone   │  │   PostgreSQL     │     │
                                       │   │  clause vec  │  │  hash-partitioned│     │
                                       │   │  retrieval   │  │  by contract_id  │     │
                                       │   └──────────────┘  └──────────────────┘     │
                                       └──────────────────────────────────────────────┘
```

### Why these choices

- **Cohere Command R**: native tool-use API, optimized for RAG + citation
  grounding, cheaper than GPT-4 for iterative agent loops.
- **Tool-based agent** (`extract_clauses`, `search_similar_clauses`,
  `score_clause_risk`): each tool is independently testable and replaceable.
- **Pinecone + Postgres hybrid**: Pinecone for sub-second semantic search,
  Postgres JSONB + GIN indexes for structured metadata filters. Pure vector
  search loses too much when the user asks "show me all NDAs from 2023".
- **Redis Sentinel over Redis Cluster**: at 500 contracts / ~15K clauses we
  don't need sharding, we need availability. Sentinel gives us that with
  one fewer moving part.
- **SSE over WebSocket**: agent output is one-way (server → client). SSE
  is plain HTTP, survives reverse proxies, and avoids the lifecycle bugs
  of long-lived WS connections.
- **Local fallbacks** for embeddings (sentence-transformers) and vector
  store (numpy on disk) so the project is clonable + runnable without any
  external accounts.

## Project structure

```
legallens/
├── docker-compose.yml          # postgres + redis master/replica + 3 sentinels
├── infra/                      # redis & sentinel configs
├── scripts/
│   ├── download_cuad.py        # fetch CUAD dataset
│   ├── ingest.py               # driver: dispatch contracts to workers
│   ├── ingest_worker.py        # worker process entrypoint
│   └── init_db.sql             # hash-partitioned schema
├── src/legallens/
│   ├── agents/loop.py          # Cohere agent loop + tool dispatch
│   ├── api/main.py             # FastAPI + SSE endpoints
│   ├── config.py               # pydantic-settings config
│   ├── coordination/           # SentinelClient, WorkerRegistry
│   ├── db.py                   # async SQLAlchemy session helper
│   ├── ingestion/              # parser, embedder, vector store
│   ├── models/domain.py        # Pydantic domain models
│   ├── orchestration/          # TaskDispatcher, pub/sub, heartbeat monitor
│   ├── prompts/tools.py        # Cohere tool definitions + system preamble
│   ├── retrieval/tools.py      # real tool implementations (PG + vector)
│   └── workers/                # ConsistentHashRing, TaskQueue, Worker
└── tests/                      # unit tests (ring, parser, ...)
```

## Setup

### Prerequisites
- Python 3.11+
- Docker (for Postgres + Redis Sentinel topology)
- *(optional)* Cohere + Pinecone API keys — without them the system falls
  back to a local sentence-transformers embedder and an on-disk numpy
  vector store.

### Quick start
```bash
# 1. Clone & install
git clone https://github.com/cristiipan/legallens.git
cd legallens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Environment
cp .env.example .env
# (Optional) fill in COHERE_API_KEY / PINECONE_API_KEY.
# Leave blank to use local fallbacks.

# 3. Start infra (Postgres + Redis master/replica + 3 Sentinels)
docker compose up -d

# 4. Download CUAD into ./data/
python scripts/download_cuad.py
# (unzip data/cuad/data.zip → data/cuad/CUAD_v1.json)

# 5. Start one or more ingest workers (separate terminals or --scale)
python scripts/ingest_worker.py &
python scripts/ingest_worker.py &

# 6. Dispatch contracts to the worker pool
python scripts/ingest.py --limit 50   # try a slice first

# 7. Start the API
uvicorn legallens.api.main:app --reload
```

### Verifying the cluster
```bash
# Hash ring tests
pytest tests/test_hash_ring.py -v

# Sentinel & master discovery
redis-cli -p 26379 sentinel get-master-addr-by-name legallens-master

# Worker heartbeats (should show one key per running worker)
redis-cli keys 'worker:hb:*'
```

## Roadmap

- [x] Project scaffold + CUAD downloader
- [x] Consistent hash ring + tests
- [x] Redis Sentinel client + worker registry + heartbeat monitor
- [x] Task dispatcher + per-worker queue + worker process
- [x] Postgres hash-partitioned schema (`contracts`, `clauses`)
- [x] Pluggable embedder (Cohere ↔ sentence-transformers)
- [x] Pluggable vector store (Pinecone ↔ local numpy)
- [x] Real tool implementations (`extract_clauses`, `search_similar_clauses`, `score_clause_risk`)
- [x] Cohere agent loop with SSE streaming endpoint
- [ ] End-to-end ingest run on the full 510-contract CUAD
- [ ] Risk scorer driven by a focused LLM call (replacing the heuristic)
- [ ] Eval suite (precision/recall on held-out CUAD subset)
- [ ] Minimal Next.js frontend
- [ ] Containerized worker image + `docker compose up --scale ingest-worker=N`

## Eval methodology (planned)

CUAD ships ground-truth annotations for 41 clause categories. Hold out 10%
of contracts and measure:
- **Clause extraction**: F1 against ground-truth spans
- **Risk scoring**: agreement with CUAD's "important clauses" labels
- **Follow-up Q&A**: faithfulness (answer cites the correct clause) + relevance

## License

MIT
