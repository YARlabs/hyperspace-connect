//! CLI entrypoint for `hyperspace-connect` — Universal Zero-ETL Bridge for HyperspaceDB.

use clap::{Parser, Subcommand};
use hyperspace_connect::{
    execute_migration, flush_stream_batch_to_hyperspace, induce_edges_by_metadata_keys,
    ingest_graph_to_hyperspace, parse_apoc_jsonl, parse_cypher_script, parse_legacy_vector_dump,
    transform_ndjson_batch, StreamEdgeRule, StreamMappingConfig, TargetManifold,
};
use hyperspace_sdk::{Client, MetadataInductionRuleProto};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "hyperspace-connect",
    version = "4.2.0",
    about = "Zero-ETL Universal Data Bridge & Migrator for HyperspaceDB (Neo4j, Kafka/NDJSON Streams, Qdrant/Pinecone/Milvus)"
)]
struct Cli {
    /// HyperspaceDB gRPC endpoint URL
    #[arg(long, env = "HYPERSPACE_URL", default_value = "http://127.0.0.1:50051")]
    url: String,

    /// Optional HyperspaceDB API key
    #[arg(long, env = "HYPERSPACE_API_KEY")]
    api_key: Option<String>,

    /// Target collection name in HyperspaceDB
    #[arg(short, long, env = "HYPERSPACE_COLLECTION", default_value = "default")]
    collection: String,

    /// Target vector dimension (default: 801 for Hybrid801 or 64 for Lorentz)
    #[arg(short, long, default_value_t = 64)]
    dimension: usize,

    /// Target manifold geometry: lorentz, poincare, hybrid801, cosine, l2
    #[arg(short, long, default_value = "lorentz")]
    manifold: String,

    /// Batch size for gRPC ingestion
    #[arg(long, default_value_t = 500)]
    batch_size: usize,

    /// Validate, parse, and compute manifold embeddings without transmitting to server
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Import nodes and relationships from Neo4j / AWS Neptune (APOC JSONL or Cypher script)
    ImportNeo4j {
        /// Path to `.jsonl` (APOC export) or `.cypher` script file
        #[arg(short, long)]
        input: PathBuf,

        /// Input format: auto, apoc, or cypher
        #[arg(long, default_value = "auto")]
        format: String,
    },
    /// Consume streaming JSON/NDJSON events (from file or stdin) and map into vectors + KAG edges
    StreamIngest {
        /// Path to NDJSON event file (or `-` for stdin)
        #[arg(short, long)]
        input: String,

        /// Field name for unique event ID
        #[arg(long, default_value = "id")]
        id_field: String,

        /// Comma-separated text fields to embed
        #[arg(long, default_value = "text,message,content")]
        text_fields: String,

        /// Declarative KAG edge rules in `field:RELATION[:weight]` format (repeatable)
        #[arg(long = "edge-rule")]
        edge_rules: Vec<String>,
    },
    /// Migrate vectors and payloads from Qdrant, Pinecone, Milvus, or Chroma JSON/JSONL exports
    MigrateVectors {
        /// Path to legacy vector dump (`.json` or `.jsonl`)
        #[arg(short, long)]
        input: PathBuf,

        /// Comma-separated metadata keys to automatically link with `SHARED_<KEY>` KAG edges
        #[arg(long, default_value = "")]
        induce_keys: String,
    },
    /// Trigger server-side declarative metadata-to-KAG edge induction (`foreign_key`, `path_hierarchy`, `exact_match`)
    InduceEdges {
        /// Rule type: foreign_key, path_hierarchy, or exact_match
        #[arg(long, default_value = "exact_match")]
        rule_type: String,

        /// Source metadata field
        #[arg(long)]
        source_field: String,

        /// Target metadata field
        #[arg(long)]
        target_field: String,

        /// Directed KAG relation label
        #[arg(long, default_value = "RELATED_TO")]
        relation: String,

        /// Edge weight
        #[arg(long, default_value_t = 1.0)]
        weight: f32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let manifold = TargetManifold::from_name(&cli.manifold);

    match cli.command {
        Commands::ImportNeo4j { input, format } => {
            let content = std::fs::read_to_string(&input)?;
            let is_cypher = match format.as_str() {
                "cypher" => true,
                "apoc" | "jsonl" => false,
                _ => input
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("cypher") || ext.eq_ignore_ascii_case("cql")),
            };

            let (nodes, edges) = if is_cypher {
                parse_cypher_script(&content)?
            } else {
                parse_apoc_jsonl(&content)?
            };

            if cli.dry_run {
                let max_depth = nodes.iter().map(|n| n.hierarchy_depth).max().unwrap_or(0);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "mode": "dry_run",
                        "command": "import-neo4j",
                        "nodes_parsed": nodes.len(),
                        "edges_parsed": edges.len(),
                        "max_hierarchy_depth": max_depth,
                        "manifold": manifold.metric_str(),
                        "dimension": cli.dimension
                    }))?
                );
                return Ok(());
            }

            let mut client = connect_and_ensure_collection(
                &cli.url,
                cli.api_key.clone(),
                &cli.collection,
                cli.dimension,
                manifold,
            )
            .await?;

            let report = ingest_graph_to_hyperspace(
                &mut client,
                &cli.collection,
                cli.dimension,
                manifold,
                &nodes,
                &edges,
                cli.batch_size,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }

        Commands::StreamIngest {
            input,
            id_field,
            text_fields,
            edge_rules,
        } => {
            let payload = if input == "-" {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&input)?
            };

            let parsed_rules: Vec<StreamEdgeRule> = edge_rules
                .iter()
                .filter_map(|r| {
                    let mut parts = r.split(':');
                    let target_field = parts.next()?.trim().to_string();
                    let relation = parts.next()?.trim().to_string();
                    let weight = parts
                        .next()
                        .and_then(|w| w.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    Some(StreamEdgeRule {
                        target_field,
                        relation,
                        weight,
                        reverse_direction: false,
                    })
                })
                .collect();

            let cfg = StreamMappingConfig {
                id_field,
                vector_field: Some("vector".to_string()),
                text_fields: text_fields
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                metadata_fields: Vec::new(),
                depth_field: Some("depth".to_string()),
                edge_rules: parsed_rules,
            };

            let batch = transform_ndjson_batch(&payload, &cfg, cli.dimension, manifold);
            if cli.dry_run {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "mode": "dry_run",
                        "command": "stream-ingest",
                        "vectors_prepared": batch.vectors.len(),
                        "edges_prepared": batch.edges.len(),
                        "skipped_events": batch.skipped_events,
                        "manifold": manifold.metric_str(),
                        "dimension": cli.dimension
                    }))?
                );
                return Ok(());
            }

            let mut client = connect_and_ensure_collection(
                &cli.url,
                cli.api_key.clone(),
                &cli.collection,
                cli.dimension,
                manifold,
            )
            .await?;

            let report =
                flush_stream_batch_to_hyperspace(&mut client, &cli.collection, batch, cli.batch_size)
                    .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }

        Commands::MigrateVectors { input, induce_keys } => {
            let content = std::fs::read_to_string(&input)?;
            let (records, dups) =
                parse_legacy_vector_dump(&content, cli.dimension, manifold)?;
            let shared_keys: Vec<String> = induce_keys
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if cli.dry_run {
                let edges = induce_edges_by_metadata_keys(&records, &shared_keys);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "mode": "dry_run",
                        "command": "migrate-vectors",
                        "records_parsed": records.len(),
                        "duplicates_removed": dups,
                        "induced_edges_prepared": edges.len(),
                        "manifold": manifold.metric_str(),
                        "dimension": cli.dimension
                    }))?
                );
                return Ok(());
            }

            let mut client = connect_and_ensure_collection(
                &cli.url,
                cli.api_key.clone(),
                &cli.collection,
                cli.dimension,
                manifold,
            )
            .await?;

            let report = execute_migration(
                &mut client,
                &cli.collection,
                records,
                dups,
                &shared_keys,
                cli.batch_size,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }

        Commands::InduceEdges {
            rule_type,
            source_field,
            target_field,
            relation,
            weight,
        } => {
            let mut client = Client::connect(cli.url, cli.api_key, None).await?;
            let rules = vec![MetadataInductionRuleProto {
                rule_type,
                field: source_field,
                target_field: Some(target_field),
                relation,
                weight: Some(weight),
            }];
            let created = client
                .induce_edges_from_metadata(rules, &cli.collection)
                .await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "collection": cli.collection,
                    "edges_created": created
                }))?
            );
        }
    }

    Ok(())
}

async fn connect_and_ensure_collection(
    url: &str,
    api_key: Option<String>,
    collection: &str,
    dimension: usize,
    manifold: TargetManifold,
) -> Result<Client, Box<dyn std::error::Error>> {
    let mut client = Client::connect(url.to_string(), api_key, None).await?;
    if !client.exists(collection.to_string()).await.unwrap_or(false) {
        let _ = client
            .create_collection_simple(
                collection.to_string(),
                dimension as u32,
                manifold.metric_str(),
            )
            .await;
    }
    Ok(client)
}
