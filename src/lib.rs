//! # `hyperspace-connect`
//!
//! Zero-ETL Universal Data Bridge & Graph/Vector Migrator for **HyperspaceDB**.
//!
//! ## Core Modules
//! - [`neo4j`]: Imports Neo4j APOC JSONL dumps and Cypher (`CREATE`/`MERGE`) scripts directly
//!   into HyperspaceDB's CSR KAG topology while computing topological hierarchy depths for
//!   Lorentz hyperbolic coordinates.
//! - [`stream`]: High-throughput streaming event processor (Kafka / Redpanda / NATS / MQTT / NDJSON)
//!   with declarative field-to-vector and field-to-KAG-edge mapping rules.
//! - [`migrator`]: One-command vector database migrator from Qdrant, Pinecone, Milvus, and Chroma
//!   with deduplication, manifold lifting, and metadata-driven KAG edge induction.
//! - [`manifold`]: Deterministic multi-manifold vector projection (`Lorentz`, `Poincare`,
//!   `Hybrid801`, `Cosine`, `Euclidean`) and collision-resistant SHA-256 ID folding.

pub mod manifold;
pub mod migrator;
pub mod neo4j;
pub mod stream;

pub use manifold::{
    embed_text_deterministic, hash_external_id_to_u32, project_vector_to_manifold, TargetManifold,
};
pub use migrator::{
    execute_migration, induce_edges_by_metadata_keys, parse_legacy_vector_dump,
    MigratedVectorRecord, VectorMigrationReport,
};
pub use neo4j::{
    ingest_graph_to_hyperspace, parse_apoc_jsonl, parse_cypher_script, GraphImportReport,
    ImportedGraphEdge, ImportedGraphNode,
};
pub use stream::{
    flush_stream_batch_to_hyperspace, transform_ndjson_batch, StreamBatchOutput, StreamEdgeRule,
    StreamIngestReport, StreamMappingConfig,
};
