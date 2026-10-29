//! Real-Time Event Stream Sink (Kafka / Redpanda / NATS / MQTT / NDJSON) to HyperspaceDB.
//!
//! Transforms streaming JSON events into multi-manifold vectors and CSR KAG edges on the fly
//! using declarative mapping rules (`StreamMappingConfig`), eliminating custom Python ETL scripts.

use crate::manifold::{
    embed_text_deterministic, hash_external_id_to_u32, project_vector_to_manifold, TargetManifold,
};
use hyperspace_sdk::{Client, DurabilityLevel, EdgeItem};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Declarative rule that maps a JSON field in an incoming event into a directed KAG edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEdgeRule {
    /// JSON field name on the event containing the target entity's ID (e.g. `"reply_to"`, `"user_id"`, `"parent_id"`).
    pub target_field: String,
    /// Directed KAG relation label (e.g. `"REPLIES_TO"`, `"AUTHORED_BY"`, `"CAUSED_BY"`).
    pub relation: String,
    /// Edge weight in `[0.0, 1.0]`.
    #[serde(default = "default_edge_weight")]
    pub weight: f32,
    /// If `true`, reverses edge direction (`target_field -> event_id` instead of `event_id -> target_field`).
    #[serde(default)]
    pub reverse_direction: bool,
}

const fn default_edge_weight() -> f32 {
    1.0
}

/// Declarative configuration for mapping streaming JSON events into HyperspaceDB vectors and KAG edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMappingConfig {
    /// Field containing the unique event or entity ID (defaults to `"id"`).
    #[serde(default = "default_id_field")]
    pub id_field: String,
    /// Optional field containing a pre-computed numeric vector array.
    #[serde(default)]
    pub vector_field: Option<String>,
    /// Text fields concatenated to generate the deterministic manifold embedding when `vector_field` is absent.
    #[serde(default)]
    pub text_fields: Vec<String>,
    /// Fields copied into HyperspaceDB point metadata (`empty` = copy all scalar fields).
    #[serde(default)]
    pub metadata_fields: Vec<String>,
    /// Optional numeric field encoding hierarchy/causal depth for Lorentzian radial scaling.
    #[serde(default)]
    pub depth_field: Option<String>,
    /// Declarative rules that emit directed KAG edges (`EdgeItem`) per event.
    #[serde(default)]
    pub edge_rules: Vec<StreamEdgeRule>,
}

fn default_id_field() -> String {
    "id".to_string()
}

impl Default for StreamMappingConfig {
    fn default() -> Self {
        Self {
            id_field: "id".to_string(),
            vector_field: Some("vector".to_string()),
            text_fields: vec![
                "text".to_string(),
                "message".to_string(),
                "content".to_string(),
            ],
            metadata_fields: Vec::new(),
            depth_field: None,
            edge_rules: Vec::new(),
        }
    }
}

/// Materialized batch of vectors and KAG edges extracted from a stream micro-batch.
#[derive(Debug, Clone, Default)]
pub struct StreamBatchOutput {
    pub vectors: Vec<(u32, Vec<f64>, HashMap<String, String>)>,
    pub edges: Vec<EdgeItem>,
    pub skipped_events: usize,
}

/// Summary report of a stream ingestion run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamIngestReport {
    pub events_processed: usize,
    pub vectors_inserted: usize,
    pub edges_linked: u64,
    pub skipped_events: usize,
}

/// Transforms a newline-delimited JSON (NDJSON) micro-batch of streaming events into
/// ready-to-insert HyperspaceDB vectors and KAG edges.
pub fn transform_ndjson_batch(
    ndjson_payload: &str,
    config: &StreamMappingConfig,
    dimension: usize,
    manifold: TargetManifold,
) -> StreamBatchOutput {
    let mut out = StreamBatchOutput::default();

    for (idx, raw_line) in ndjson_payload.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<Value>(line) else {
            out.skipped_events += 1;
            continue;
        };
        let Some(obj) = val.as_object() else {
            out.skipped_events += 1;
            continue;
        };

        let ext_id = obj
            .get(&config.id_field)
            .and_then(scalar_to_string)
            .unwrap_or_else(|| format!("stream_evt_{idx}"));
        let hs_id = hash_external_id_to_u32(&ext_id);

        let depth = config
            .depth_field
            .as_ref()
            .and_then(|df| obj.get(df))
            .and_then(Value::as_u64)
            .map_or(1_u32, |d| d as u32);

        // Extract or compute manifold vector
        let raw_vec = config
            .vector_field
            .as_ref()
            .and_then(|vf| obj.get(vf))
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_f64).collect::<Vec<f64>>())
            .filter(|v| !v.is_empty());

        let vector = if let Some(raw) = raw_vec {
            project_vector_to_manifold(&raw, dimension, manifold)
        } else {
            let mut text_parts = Vec::new();
            if config.text_fields.is_empty() {
                for (k, v) in obj {
                    if let Some(s) = scalar_to_string(v) {
                        text_parts.push(format!("{k}:{s}"));
                    }
                }
            } else {
                for tf in &config.text_fields {
                    if let Some(v) = obj.get(tf).and_then(scalar_to_string) {
                        if !v.is_empty() {
                            text_parts.push(v);
                        }
                    }
                }
            }
            let combined_text = if text_parts.is_empty() {
                ext_id.clone()
            } else {
                text_parts.join(" ")
            };
            embed_text_deterministic(&combined_text, depth, dimension, manifold)
        };

        // Extract metadata
        let mut meta = HashMap::new();
        meta.insert("_ext_id".to_string(), ext_id.clone());
        if config.metadata_fields.is_empty() {
            for (k, v) in obj {
                if config.vector_field.as_deref() == Some(k.as_str()) {
                    continue;
                }
                if let Some(s) = scalar_to_string(v) {
                    meta.insert(k.clone(), s);
                }
            }
        } else {
            for mf in &config.metadata_fields {
                if let Some(s) = obj.get(mf).and_then(scalar_to_string) {
                    meta.insert(mf.clone(), s);
                }
            }
        }

        out.vectors.push((hs_id, vector, meta));

        // Evaluate declarative KAG edge rules
        for rule in &config.edge_rules {
            if let Some(target_val) = obj.get(&rule.target_field) {
                let target_ids: Vec<String> = match target_val {
                    Value::Array(arr) => arr.iter().filter_map(scalar_to_string).collect(),
                    other => scalar_to_string(other).into_iter().collect(),
                };
                for target_ext in target_ids {
                    if target_ext.is_empty() {
                        continue;
                    }
                    let target_hs_id = hash_external_id_to_u32(&target_ext);
                    if target_hs_id == hs_id {
                        continue;
                    }
                    let (src, dst) = if rule.reverse_direction {
                        (target_hs_id, hs_id)
                    } else {
                        (hs_id, target_hs_id)
                    };
                    out.edges.push(EdgeItem {
                        source_id: src,
                        target_id: dst,
                        relation: rule.relation.clone(),
                        weight: rule.weight,
                    });
                }
            }
        }
    }

    out
}

/// Flushes a transformed stream batch into HyperspaceDB via `batch_insert` and `link_batch`.
pub async fn flush_stream_batch_to_hyperspace(
    client: &mut Client,
    collection: &str,
    batch: StreamBatchOutput,
    chunk_size: usize,
) -> Result<StreamIngestReport, String> {
    let step = chunk_size.max(1);
    let mut vectors_inserted = 0usize;

    for chunk in batch.vectors.chunks(step) {
        client
            .batch_insert(
                chunk.to_vec(),
                Some(collection.to_string()),
                DurabilityLevel::default(),
            )
            .await
            .map_err(|e| format!("Stream batch_insert failed: {e}"))?;
        vectors_inserted += chunk.len();
    }

    let mut edges_linked = 0_u64;
    for chunk in batch.edges.chunks(step) {
        let linked = client
            .link_batch(chunk.to_vec(), collection)
            .await
            .map_err(|e| format!("Stream link_batch failed: {e}"))?;
        edges_linked += linked;
    }

    Ok(StreamIngestReport {
        events_processed: vectors_inserted + batch.skipped_events,
        vectors_inserted,
        edges_linked,
        skipped_events: batch.skipped_events,
    })
}

fn scalar_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_transform_with_edge_rules() {
        let config = StreamMappingConfig {
            id_field: "msg_id".to_string(),
            vector_field: None,
            text_fields: vec!["message".to_string()],
            metadata_fields: vec!["user_id".to_string(), "channel".to_string()],
            depth_field: Some("thread_depth".to_string()),
            edge_rules: vec![
                StreamEdgeRule {
                    target_field: "reply_to".to_string(),
                    relation: "REPLIES_TO".to_string(),
                    weight: 0.95,
                    reverse_direction: false,
                },
                StreamEdgeRule {
                    target_field: "user_id".to_string(),
                    relation: "AUTHORED_BY".to_string(),
                    weight: 1.0,
                    reverse_direction: false,
                },
            ],
        };

        let events = r#"
{"msg_id":"m_100","user_id":"u_alice","channel":"robotics","message":"Telemetry nominal on drone #4","thread_depth":0}
{"msg_id":"m_101","user_id":"u_bob","channel":"robotics","message":"Confirmed, Lorentz trajectory converged","reply_to":"m_100","thread_depth":1}
"#;

        let out = transform_ndjson_batch(events, &config, 64, TargetManifold::Lorentz);
        assert_eq!(out.vectors.len(), 2);
        assert_eq!(out.skipped_events, 0);
        // Event 1 emits AUTHORED_BY; Event 2 emits REPLIES_TO + AUTHORED_BY => 3 edges total
        assert_eq!(out.edges.len(), 3);
        assert!(out.edges.iter().any(|e| e.relation == "REPLIES_TO"));
        assert!(out.edges.iter().any(|e| e.relation == "AUTHORED_BY"));
    }
}
