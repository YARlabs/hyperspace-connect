//! Legacy Vector Database Migrator (Qdrant, Pinecone, Milvus, Chroma) to HyperspaceDB.
//!
//! Parses vector and payload exports from Qdrant (`result.points`), Pinecone (`vectors` / `matches`),
//! Milvus, and Chroma, deduplicates point IDs, projects embeddings into the target HyperspaceDB
//! manifold (Cosine, Euclidean, Lorentz, Poincaré, or 801D Hybrid MRL), and optionally induces
//! CSR KAG edges across shared metadata keys.

use crate::manifold::{hash_external_id_to_u32, project_vector_to_manifold, TargetManifold};
use hyperspace_sdk::{Client, DurabilityLevel, EdgeItem};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// A normalized vector record extracted from Qdrant, Pinecone, Milvus, or Chroma.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigratedVectorRecord {
    pub external_id: String,
    pub hs_id: u32,
    pub vector: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// Summary report of a legacy vector database migration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorMigrationReport {
    pub total_records_parsed: usize,
    pub duplicates_removed: usize,
    pub vectors_inserted: usize,
    pub induced_edges_linked: u64,
}

/// Parses a JSON or JSONL export from Qdrant, Pinecone, Milvus, or Chroma into normalized
/// [`MigratedVectorRecord`]s with automatic deduplication and manifold projection.
pub fn parse_legacy_vector_dump(
    payload: &str,
    target_dimension: usize,
    manifold: TargetManifold,
) -> Result<(Vec<MigratedVectorRecord>, usize), String> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let mut raw_objects = Vec::new();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(root) = serde_json::from_str::<Value>(trimmed) {
            extract_records_from_json_tree(&root, &mut raw_objects);
        }
    }

    // Fallback to line-by-line JSONL if root wasn't a single JSON document or yielded 0 items
    if raw_objects.is_empty() {
        for (line_idx, raw_line) in trimmed.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let val: Value = serde_json::from_str(line)
                .map_err(|e| format!("Invalid JSONL at line {}: {e}", line_idx + 1))?;
            extract_records_from_json_tree(&val, &mut raw_objects);
        }
    }

    let mut seen_ids = HashSet::new();
    let mut records = Vec::with_capacity(raw_objects.len());
    let mut duplicates = 0usize;

    for (idx, item) in raw_objects.into_iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };

        let ext_id = obj
            .get("id")
            .or_else(|| obj.get("point_id"))
            .or_else(|| obj.get("_id"))
            .and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| format!("migrated_{idx}"));

        let hs_id = hash_external_id_to_u32(&ext_id);
        if !seen_ids.insert(hs_id) {
            duplicates += 1;
            continue;
        }

        // Extract vector from Qdrant (`vector`), Pinecone (`values`), Milvus/Chroma (`embedding` / `embeddings`)
        let raw_vec = obj
            .get("vector")
            .or_else(|| obj.get("values"))
            .or_else(|| obj.get("embedding"))
            .or_else(|| obj.get("embeddings"))
            .and_then(extract_float_vec)
            .ok_or_else(|| format!("Record '{ext_id}' is missing a valid numeric vector field"))?;

        let effective_dim = if target_dimension == 0 {
            raw_vec.len()
        } else {
            target_dimension
        };
        let projected = project_vector_to_manifold(&raw_vec, effective_dim, manifold);

        // Extract metadata from Qdrant (`payload`), Pinecone/Chroma (`metadata`), or top-level scalar fields
        let mut metadata = HashMap::new();
        metadata.insert("_source_id".to_string(), ext_id.clone());

        if let Some(meta_obj) = obj
            .get("payload")
            .or_else(|| obj.get("metadata"))
            .and_then(Value::as_object)
        {
            for (k, v) in meta_obj {
                if let Some(s) = scalar_to_str(v) {
                    metadata.insert(k.clone(), s);
                }
            }
        } else {
            for (k, v) in obj {
                if matches!(
                    k.as_str(),
                    "id" | "point_id" | "_id" | "vector" | "values" | "embedding" | "embeddings"
                ) {
                    continue;
                }
                if let Some(s) = scalar_to_str(v) {
                    metadata.insert(k.clone(), s);
                }
            }
        }

        records.push(MigratedVectorRecord {
            external_id: ext_id,
            hs_id,
            vector: projected,
            metadata,
        });
    }

    Ok((records, duplicates))
}

/// Induces directed KAG edges (`SAME_<KEY>`) between migrated records that share an identical
/// metadata value for any key in `shared_metadata_keys`.
#[must_use]
pub fn induce_edges_by_metadata_keys(
    records: &[MigratedVectorRecord],
    shared_metadata_keys: &[String],
) -> Vec<EdgeItem> {
    let mut edges = Vec::new();
    for key in shared_metadata_keys {
        let mut groups: HashMap<&str, Vec<u32>> = HashMap::new();
        for rec in records {
            if let Some(val) = rec.metadata.get(key) {
                if !val.is_empty() {
                    groups.entry(val.as_str()).or_default().push(rec.hs_id);
                }
            }
        }
        let relation = format!("SHARED_{}", key.to_ascii_uppercase());
        for (_, ids) in groups {
            if ids.len() < 2 {
                continue;
            }
            // Connect hub (first node in group) to members to avoid O(N^2) clique explosion
            let hub = ids[0];
            for &member in &ids[1..] {
                if hub != member {
                    edges.push(EdgeItem {
                        source_id: hub,
                        target_id: member,
                        relation: relation.clone(),
                        weight: 0.85,
                    });
                }
            }
        }
    }
    edges
}

/// Uploads migrated vector records and optional induced KAG edges into HyperspaceDB.
pub async fn execute_migration(
    client: &mut Client,
    collection: &str,
    records: Vec<MigratedVectorRecord>,
    duplicates_removed: usize,
    shared_metadata_keys: &[String],
    batch_size: usize,
) -> Result<VectorMigrationReport, String> {
    let step = batch_size.max(1);
    let total_parsed = records.len() + duplicates_removed;
    let induced_edges = induce_edges_by_metadata_keys(&records, shared_metadata_keys);

    let items: Vec<(u32, Vec<f64>, HashMap<String, String>)> = records
        .into_iter()
        .map(|r| (r.hs_id, r.vector, r.metadata))
        .collect();

    let mut vectors_inserted = 0usize;
    for chunk in items.chunks(step) {
        client
            .batch_insert(
                chunk.to_vec(),
                Some(collection.to_string()),
                DurabilityLevel::default(),
            )
            .await
            .map_err(|e| format!("Migration batch_insert failed: {e}"))?;
        vectors_inserted += chunk.len();
    }

    let mut induced_edges_linked = 0_u64;
    for chunk in induced_edges.chunks(step) {
        let linked = client
            .link_batch(chunk.to_vec(), collection)
            .await
            .map_err(|e| format!("Migration link_batch failed: {e}"))?;
        induced_edges_linked += linked;
    }

    Ok(VectorMigrationReport {
        total_records_parsed: total_parsed,
        duplicates_removed,
        vectors_inserted,
        induced_edges_linked,
    })
}

fn extract_records_from_json_tree(val: &Value, out: &mut Vec<Value>) {
    match val {
        Value::Array(arr) => {
            for item in arr {
                out.push(item.clone());
            }
        }
        Value::Object(obj) => {
            // Qdrant scroll format: { "result": { "points": [...] } }
            if let Some(points) = obj
                .get("result")
                .and_then(|r| r.get("points"))
                .and_then(Value::as_array)
            {
                for p in points {
                    out.push(p.clone());
                }
                return;
            }
            // Pinecone / Qdrant list format: { "vectors": [...] } or { "points": [...] } or { "matches": [...] }
            for key in ["points", "vectors", "matches", "records", "data"] {
                if let Some(arr) = obj.get(key).and_then(Value::as_array) {
                    for p in arr {
                        out.push(p.clone());
                    }
                    return;
                }
                // Pinecone fetch format: { "vectors": { "id1": { "id": "id1", "values": [...] } } }
                if let Some(map) = obj.get(key).and_then(Value::as_object) {
                    for (k, v) in map {
                        let mut cloned = v.clone();
                        if let Some(inner_obj) = cloned.as_object_mut() {
                            inner_obj
                                .entry("id".to_string())
                                .or_insert_with(|| Value::String(k.clone()));
                        }
                        out.push(cloned);
                    }
                    return;
                }
            }
            out.push(val.clone());
        }
        _ => {}
    }
}

fn extract_float_vec(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Array(arr) => {
            let vec: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
            if vec.is_empty() {
                None
            } else {
                Some(vec)
            }
        }
        // Named vectors in Qdrant: { "default": [0.1, 0.2, ...] }
        Value::Object(map) => {
            for (_, inner) in map {
                if let Some(vec) = extract_float_vec(inner) {
                    return Some(vec);
                }
            }
            None
        }
        _ => None,
    }
}

fn scalar_to_str(v: &Value) -> Option<String> {
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
    fn test_parse_qdrant_and_pinecone_exports_with_dedup() {
        let qdrant_json = r#"
{
  "result": {
    "points": [
      {"id": "doc_1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"category": "finance", "author": "alice"}},
      {"id": "doc_2", "vector": [0.5, 0.6, 0.7, 0.8], "payload": {"category": "finance", "author": "bob"}},
      {"id": "doc_1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"category": "finance", "author": "alice"}}
    ]
  }
}
"#;
        let (records, dups) =
            parse_legacy_vector_dump(qdrant_json, 8, TargetManifold::Lorentz).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(dups, 1);
        assert_eq!(records[0].vector.len(), 8);

        let edges = induce_edges_by_metadata_keys(&records, &["category".to_string()]);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, "SHARED_CATEGORY");
    }
}
