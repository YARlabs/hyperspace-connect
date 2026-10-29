# `hyperspace-connect` — Zero-ETL Universal Data Bridge & Migrator for HyperspaceDB

[![Crates.io](https://img.shields.io/crates/v/hyperspace-connect.svg)](https://crates.io/crates/hyperspace-connect)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

`hyperspace-connect` is a standalone, zero-overhead Rust CLI daemon and library that bridges **Neo4j / AWS Neptune**, **Kafka / Redpanda / MQTT event streams**, and **legacy vector databases (Qdrant, Pinecone, Milvus, Chroma)** directly into **[HyperspaceDB](https://yar.ink)** without custom Python ETL scripts.

It depends exclusively on the published [`hyperspace-sdk`](https://crates.io/crates/hyperspace-sdk) (`v4.2.0`) gRPC client.

---

## Key Capabilities

1. **Neo4j & Cypher Graph Importer (`import-neo4j`)**
   - Parses Neo4j APOC JSONL exports (`apoc.export.json.all`) and Cypher `CREATE` / `MERGE` scripts.
   - Automatically computes topological hierarchy depths (`root -> leaf`) and embeds nodes into **Lorentz Hyperboloid** ($\mathbb{H}^d$) or **801D Hybrid MRL** space so root concepts naturally reside near the hyperboloid apex ($x_0 \approx 1$).
   - Compresses relationships into HyperspaceDB's lock-free CSR KAG topology via `link_batch` (reducing memory footprint by 50–80x compared to JVM graph stores).

2. **Real-Time Event Stream Sink (`stream-ingest`)**
   - Consumes NDJSON streams from Kafka (`kcat`), Redpanda, NATS, or MQTT pipes.
   - Evaluates declarative `--edge-rule field:RELATION:weight` mappings on the fly (e.g., `reply_to:REPLIES_TO:0.95`, `user_id:AUTHORED_BY:1.0`) to construct real-time episodic agent memory graphs.

3. **One-Command Legacy Vector Migrator (`migrate-vectors`)**
   - Ingests point dumps from **Qdrant**, **Pinecone**, **Milvus**, and **Chroma**.
   - Performs deterministic SHA-256 ID deduplication, lifts Euclidean/Cosine vectors onto hyperbolic or 801D Hybrid manifolds, and induces `SHARED_<KEY>` KAG edges across shared metadata fields.

---

## Installation

```bash
cargo install hyperspace-connect
```

---

## Quick Start CLI Usage

### 1. Import a Neo4j APOC JSONL or Cypher Graph
```bash
hyperspace-connect \
  --url http://127.0.0.1:50051 \
  --collection enterprise_kg \
  --manifold lorentz \
  --dimension 64 \
  import-neo4j --input ./neo4j_export.jsonl
```

### 2. Stream Kafka / Redpanda Events into HyperspaceDB KAG
```bash
kcat -b localhost:9092 -t agent_events -C -e | \
  hyperspace-connect \
    --collection episodic_memory \
    --manifold hybrid801 \
    --dimension 801 \
    stream-ingest \
    --input - \
    --id-field msg_id \
    --text-fields "message,summary" \
    --edge-rule "reply_to:REPLIES_TO:0.95" \
    --edge-rule "user_id:AUTHORED_BY:1.0"
```

### 3. Migrate from Qdrant / Pinecone with Automatic KAG Edge Induction
```bash
hyperspace-connect \
  --collection migrated_docs \
  --manifold lorentz \
  --dimension 128 \
  migrate-vectors \
  --input ./qdrant_scroll_dump.json \
  --induce-keys "category,department"
```

---

## License

Apache-2.0. Copyright © 2026 **YARlabs**.

