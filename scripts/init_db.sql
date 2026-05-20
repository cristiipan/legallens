-- LegalLens schema.
--
-- The two largest tables (contracts, clauses) are partitioned by HASH(contract_id).
-- The hash partition strategy is what makes parallel ingestion safe: each worker
-- owns a partition slice (chosen via the consistent-hash ring), so two workers
-- never contend on the same contract's rows.
--
-- 8 partitions. The exact number is somewhat arbitrary; what matters is that
-- it exceeds the steady-state worker count so the ring can rebalance without
-- reshuffling every contract.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- contracts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS contracts (
    contract_id   TEXT        NOT NULL,
    filename      TEXT        NOT NULL,
    contract_type TEXT,
    parties       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    effective_date DATE,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id)
) PARTITION BY HASH (contract_id);

CREATE TABLE IF NOT EXISTS contracts_p0 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE IF NOT EXISTS contracts_p1 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE IF NOT EXISTS contracts_p2 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE IF NOT EXISTS contracts_p3 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE IF NOT EXISTS contracts_p4 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE IF NOT EXISTS contracts_p5 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE IF NOT EXISTS contracts_p6 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE IF NOT EXISTS contracts_p7 PARTITION OF contracts FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE INDEX IF NOT EXISTS contracts_type_idx ON contracts (contract_type);
CREATE INDEX IF NOT EXISTS contracts_metadata_idx ON contracts USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- clauses
-- ---------------------------------------------------------------------------
-- Partitioned on contract_id (not clause_id) so all clauses of one contract
-- land in the same partition. Joins between contracts and clauses become
-- partition-local, and the dispatcher can route every task for a given
-- contract to a single worker.

CREATE TABLE IF NOT EXISTS clauses (
    clause_id      TEXT        NOT NULL,
    contract_id    TEXT        NOT NULL,
    category       TEXT        NOT NULL,
    text           TEXT        NOT NULL,
    page           INT,
    risk_level     TEXT,
    risk_rationale TEXT,
    vector_id      TEXT,
    embedded_at    TIMESTAMPTZ,
    PRIMARY KEY (contract_id, clause_id)
) PARTITION BY HASH (contract_id);

CREATE TABLE IF NOT EXISTS clauses_p0 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE IF NOT EXISTS clauses_p1 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE IF NOT EXISTS clauses_p2 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE IF NOT EXISTS clauses_p3 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE IF NOT EXISTS clauses_p4 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE IF NOT EXISTS clauses_p5 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE IF NOT EXISTS clauses_p6 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE IF NOT EXISTS clauses_p7 PARTITION OF clauses FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE INDEX IF NOT EXISTS clauses_category_idx ON clauses (contract_id, category);
CREATE INDEX IF NOT EXISTS clauses_vector_id_idx ON clauses (vector_id);

-- ---------------------------------------------------------------------------
-- ingest_tasks
-- ---------------------------------------------------------------------------
-- Audit trail for the distributed ingest pipeline. Helps debug worker
-- assignments and lets a re-run skip contracts already done.

CREATE TABLE IF NOT EXISTS ingest_tasks (
    task_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT        NOT NULL,
    worker_id   TEXT        NOT NULL,
    status      TEXT        NOT NULL CHECK (status IN ('queued','running','done','failed')),
    error       TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ingest_tasks_contract_idx ON ingest_tasks (contract_id);
CREATE INDEX IF NOT EXISTS ingest_tasks_worker_idx ON ingest_tasks (worker_id);
