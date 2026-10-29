//! Neo4j / AWS Neptune (APOC JSONL & Cypher Script) Importer to HyperspaceDB CSR KAG Graph.
//!
//! Parses graph nodes, labels, properties, and directed relationships from Neo4j APOC JSONL
//! exports (`apoc.export.json.all`) or Cypher `CREATE`/`MERGE` scripts, computes topological
//! hierarchy depths, embeds nodes into Lorentzian or 801D Hybrid MRL coordinates, and streams
//! vector batches and `EdgeItem` batches into HyperspaceDB.

use crate::manifold::{
    embed_text_deterministic, hash_external_id_to_u32, project_vector_to_manifold, TargetManifold,
};
use hyperspace_sdk::{Client, DurabilityLevel, EdgeItem};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

/// Parsed graph node ready for HyperspaceDB vector and KAG ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedGraphNode {
    /// Original string/numeric ID from Neo4j or Cypher variable name.
    pub external_id: String,
    /// Deterministic `u32` node ID in HyperspaceDB.
    pub hs_id: u32,
    /// Node labels (e.g. `["Person", "Developer"]`).
    pub labels: Vec<String>,
    /// Flattened string metadata for HyperspaceDB payload/filtering.
    pub properties: HashMap<String, String>,
    /// Optional pre-existing embedding vector extracted from node properties.
    pub raw_vector: Option<Vec<f64>>,
    /// Topological depth from root nodes in the graph (used for Lorentz radial coordinate).
    pub hierarchy_depth: u32,
}

/// Parsed directed relationship between two graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedGraphEdge {
    pub source_external_id: String,
    pub target_external_id: String,
    pub source_hs_id: u32,
    pub target_hs_id: u32,
    pub relation: String,
    pub weight: f32,
}

/// Summary statistics of a Neo4j / Cypher graph import operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphImportReport {
    pub nodes_parsed: usize,
    pub edges_parsed: usize,
    pub nodes_inserted: usize,
    pub edges_linked: u64,
    pub max_hierarchy_depth: u32,
}

/// Parses a Neo4j APOC JSONL stream (`{"type":"node",...}` and `{"type":"relationship",...}`)
/// into structured nodes and directed edges with computed topological hierarchy depths.
pub fn parse_apoc_jsonl(content: &str) -> Result<(Vec<ImportedGraphNode>, Vec<ImportedGraphEdge>), String> {
    let mut nodes_by_ext: HashMap<String, ImportedGraphNode> = HashMap::new();
    let mut edges = Vec::new();

    for (line_idx, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        let val: Value = serde_json::from_str(line)
            .map_err(|e| format!("Invalid APOC JSON at line {}: {e}", line_idx + 1))?;

        let record_type = val
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("node");

        match record_type {
            "node" => {
                let ext_id = extract_id_field(&val, "id")
                    .ok_or_else(|| format!("Missing node 'id' at line {}", line_idx + 1))?;
                let hs_id = hash_external_id_to_u32(&ext_id);

                let labels = val
                    .get("labels")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let mut properties = HashMap::new();
                properties.insert("_neo4j_id".to_string(), ext_id.clone());
                if !labels.is_empty() {
                    properties.insert("_labels".to_string(), labels.join(":"));
                }

                let mut raw_vector = None;
                if let Some(props_obj) = val.get("properties").and_then(Value::as_object) {
                    for (k, v) in props_obj {
                        if matches!(k.as_str(), "embedding" | "vector" | "embeddings") {
                            if let Some(arr) = v.as_array() {
                                let parsed_vec: Vec<f64> =
                                    arr.iter().filter_map(Value::as_f64).collect();
                                if !parsed_vec.is_empty() {
                                    raw_vector = Some(parsed_vec);
                                    continue;
                                }
                            }
                        }
                        properties.insert(k.clone(), json_value_to_meta_string(v));
                    }
                }

                nodes_by_ext.insert(
                    ext_id.clone(),
                    ImportedGraphNode {
                        external_id: ext_id,
                        hs_id,
                        labels,
                        properties,
                        raw_vector,
                        hierarchy_depth: 0,
                    },
                );
            }
            "relationship" | "edge" => {
                let src_ext = val
                    .get("start")
                    .and_then(|s| extract_id_field(s, "id"))
                    .or_else(|| extract_id_field(&val, "source"))
                    .or_else(|| extract_id_field(&val, "start_id"))
                    .ok_or_else(|| format!("Missing relationship start ID at line {}", line_idx + 1))?;

                let dst_ext = val
                    .get("end")
                    .and_then(|e| extract_id_field(e, "id"))
                    .or_else(|| extract_id_field(&val, "target"))
                    .or_else(|| extract_id_field(&val, "end_id"))
                    .ok_or_else(|| format!("Missing relationship end ID at line {}", line_idx + 1))?;

                let relation = val
                    .get("label")
                    .or_else(|| val.get("relation"))
                    .or_else(|| val.get("rel_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("RELATED_TO")
                    .to_string();

                let weight = val
                    .get("properties")
                    .and_then(|p| p.get("weight"))
                    .or_else(|| val.get("weight"))
                    .and_then(Value::as_f64)
                    .map_or(1.0_f32, |w| w as f32);

                let source_hs_id = nodes_by_ext
                    .get(&src_ext)
                    .map_or_else(|| hash_external_id_to_u32(&src_ext), |n| n.hs_id);
                let target_hs_id = nodes_by_ext
                    .get(&dst_ext)
                    .map_or_else(|| hash_external_id_to_u32(&dst_ext), |n| n.hs_id);

                edges.push(ImportedGraphEdge {
                    source_external_id: src_ext,
                    target_external_id: dst_ext,
                    source_hs_id,
                    target_hs_id,
                    relation,
                    weight,
                });
            }
            other => {
                return Err(format!("Unsupported APOC record type '{other}' at line {}", line_idx + 1));
            }
        }
    }

    let mut nodes: Vec<ImportedGraphNode> = nodes_by_ext.into_values().collect();
    nodes.sort_by(|a, b| a.external_id.cmp(&b.external_id));
    compute_hierarchy_depths(&mut nodes, &edges);

    Ok((nodes, edges))
}

/// Parses Cypher `CREATE` / `MERGE` statements into nodes and directed edges.
///
/// Supports patterns like:
/// - `CREATE (a:Person {name: "Alice", role: "Researcher"})`
/// - `CREATE (a)-[:REPORTS_TO {weight: 0.95}]->(b)`
pub fn parse_cypher_script(script: &str) -> Result<(Vec<ImportedGraphNode>, Vec<ImportedGraphEdge>), String> {
    let mut nodes_by_var: HashMap<String, ImportedGraphNode> = HashMap::new();
    let mut edges = Vec::new();

    for raw_line in script.lines() {
        let line = raw_line.trim().trim_end_matches(';').trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("--") {
            continue;
        }
        let body = line
            .strip_prefix("CREATE ")
            .or_else(|| line.strip_prefix("create "))
            .or_else(|| line.strip_prefix("MERGE "))
            .or_else(|| line.strip_prefix("merge "))
            .unwrap_or(line)
            .trim();

        // Check if this is an edge pattern: (src)-[:REL ...]->(dst)
        if let Some((left_part, right_part)) = body.split_once("]->(") {
            if let Some((src_part, rel_part)) = left_part.split_once(")-[") {
                let src_var = src_part
                    .trim()
                    .trim_start_matches('(')
                    .split([':', ' ', '{'])
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let dst_var = right_part
                    .trim()
                    .trim_end_matches(')')
                    .split([':', ' ', '{'])
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();

                let rel_clean = rel_part.trim().trim_end_matches(']');
                let relation = rel_clean
                    .trim_start_matches(':')
                    .split(|c: char| c.is_whitespace() || c == '{')
                    .next()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("RELATED_TO")
                    .to_string();

                let props = parse_cypher_inline_props(rel_clean);
                let weight = props
                    .get("weight")
                    .and_then(|w| w.parse::<f32>().ok())
                    .unwrap_or(1.0);

                if !src_var.is_empty() && !dst_var.is_empty() {
                    let source_hs_id = nodes_by_var
                        .get(&src_var)
                        .map_or_else(|| hash_external_id_to_u32(&src_var), |n| n.hs_id);
                    let target_hs_id = nodes_by_var
                        .get(&dst_var)
                        .map_or_else(|| hash_external_id_to_u32(&dst_var), |n| n.hs_id);

                    edges.push(ImportedGraphEdge {
                        source_external_id: src_var,
                        target_external_id: dst_var,
                        source_hs_id,
                        target_hs_id,
                        relation,
                        weight,
                    });
                }
                continue;
            }
        }

        // Node pattern: (var:Label1:Label2 {k: "v", ...})
        if body.starts_with('(') && body.ends_with(')') {
            let inner = &body[1..body.len() - 1];
            let header = inner.split('{').next().unwrap_or(inner).trim();
            let mut parts = header.split(':').map(str::trim);
            let var_name = parts.next().unwrap_or("").to_string();
            let labels: Vec<String> = parts
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect();

            let mut props = parse_cypher_inline_props(inner);
            let ext_id = props
                .get("id")
                .cloned()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| var_name.clone());

            if ext_id.is_empty() {
                continue;
            }
            if !labels.is_empty() {
                props.insert("_labels".to_string(), labels.join(":"));
            }
            props.insert("_cypher_var".to_string(), var_name.clone());

            let hs_id = hash_external_id_to_u32(&ext_id);
            let node = ImportedGraphNode {
                external_id: ext_id.clone(),
                hs_id,
                labels,
                properties: props,
                raw_vector: None,
                hierarchy_depth: 0,
            };
            if !var_name.is_empty() {
                nodes_by_var.insert(var_name, node);
            } else {
                nodes_by_var.insert(ext_id, node);
            }
        }
    }

    let mut nodes: Vec<ImportedGraphNode> = nodes_by_var.into_values().collect();
    nodes.sort_by(|a, b| a.external_id.cmp(&b.external_id));
    compute_hierarchy_depths(&mut nodes, &edges);
    Ok((nodes, edges))
}

/// Materializes vectors for all parsed graph nodes in the target manifold and ingests
/// both nodes and directed edges into HyperspaceDB in batches.
pub async fn ingest_graph_to_hyperspace(
    client: &mut Client,
    collection: &str,
    dimension: usize,
    manifold: TargetManifold,
    nodes: &[ImportedGraphNode],
    edges: &[ImportedGraphEdge],
    batch_size: usize,
) -> Result<GraphImportReport, String> {
    let effective_batch = batch_size.max(1);
    let mut vectors_to_insert = Vec::with_capacity(nodes.len());
    let mut max_depth = 0_u32;

    for node in nodes {
        max_depth = max_depth.max(node.hierarchy_depth);
        let vec = if let Some(ref raw) = node.raw_vector {
            project_vector_to_manifold(raw, dimension, manifold)
        } else {
            let mut text_buf = node.labels.join(" ");
            for (k, v) in &node.properties {
                if !k.starts_with('_') {
                    text_buf.push(' ');
                    text_buf.push_str(k);
                    text_buf.push(':');
                    text_buf.push_str(v);
                }
            }
            embed_text_deterministic(&text_buf, node.hierarchy_depth, dimension, manifold)
        };

        let mut meta = node.properties.clone();
        meta.insert(
            "_hierarchy_depth".to_string(),
            node.hierarchy_depth.to_string(),
        );
        vectors_to_insert.push((node.hs_id, vec, meta));
    }

    let mut nodes_inserted = 0usize;
    for chunk in vectors_to_insert.chunks(effective_batch) {
        client
            .batch_insert(
                chunk.to_vec(),
                Some(collection.to_string()),
                DurabilityLevel::default(),
            )
            .await
            .map_err(|e| format!("HyperspaceDB batch_insert failed: {e}"))?;
        nodes_inserted += chunk.len();
    }

    let edge_items: Vec<EdgeItem> = edges
        .iter()
        .map(|e| EdgeItem {
            source_id: e.source_hs_id,
            target_id: e.target_hs_id,
            relation: e.relation.clone(),
            weight: e.weight,
        })
        .collect();

    let mut edges_linked = 0_u64;
    for chunk in edge_items.chunks(effective_batch) {
        let linked = client
            .link_batch(chunk.to_vec(), collection)
            .await
            .map_err(|e| format!("HyperspaceDB link_batch failed: {e}"))?;
        edges_linked += linked;
    }

    Ok(GraphImportReport {
        nodes_parsed: nodes.len(),
        edges_parsed: edges.len(),
        nodes_inserted,
        edges_linked,
        max_hierarchy_depth: max_depth,
    })
}

fn compute_hierarchy_depths(nodes: &mut [ImportedGraphNode], edges: &[ImportedGraphEdge]) {
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut in_deg: HashMap<u32, usize> = HashMap::new();

    for n in nodes.iter() {
        in_deg.insert(n.hs_id, 0);
    }
    for e in edges {
        if e.source_hs_id != e.target_hs_id {
            adj.entry(e.source_hs_id).or_default().push(e.target_hs_id);
            *in_deg.entry(e.target_hs_id).or_insert(0) += 1;
        }
    }

    let mut depths: HashMap<u32, u32> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    for (&id, &deg) in &in_deg {
        if deg == 0 {
            depths.insert(id, 0);
            queue.push_back(id);
        }
    }

    // If the graph is purely cyclic (no in-degree 0 roots), seed with the first node
    if queue.is_empty() {
        if let Some(first) = nodes.first() {
            depths.insert(first.hs_id, 0);
            queue.push_back(first.hs_id);
        }
    }

    while let Some(curr) = queue.pop_front() {
        let d = depths.get(&curr).copied().unwrap_or(0);
        if let Some(children) = adj.get(&curr) {
            for &child in children {
                if let std::collections::hash_map::Entry::Vacant(e) = depths.entry(child) {
                    e.insert(d.saturating_add(1));
                    queue.push_back(child);
                }
            }
        }
    }

    for n in nodes.iter_mut() {
        n.hierarchy_depth = depths.get(&n.hs_id).copied().unwrap_or(1);
    }
}

fn extract_id_field(val: &Value, key: &str) -> Option<String> {
    match val.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn json_value_to_meta_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn parse_cypher_inline_props(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some((_, after_brace)) = text.split_once('{') {
        if let Some((inside, _)) = after_brace.rsplit_once('}') {
            for pair in inside.split(',') {
                if let Some((k, v)) = pair.split_once(':') {
                    let key = k.trim().trim_matches('"').trim_matches('\'').to_string();
                    let val = v.trim().trim_matches('"').trim_matches('\'').to_string();
                    if !key.is_empty() {
                        map.insert(key, val);
                    }
                }
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_apoc_jsonl_and_depths() {
        let jsonl = r#"
{"type":"node","id":"100","labels":["Company"],"properties":{"name":"YARlabs","sector":"AI"}}
{"type":"node","id":"101","labels":["Department"],"properties":{"name":"CoreDB"}}
{"type":"node","id":"102","labels":["Engineer"],"properties":{"name":"Alice","role":"Rust"}}
{"type":"relationship","id":"r1","label":"HAS_DEPT","start":{"id":"100"},"end":{"id":"101"},"properties":{"weight":1.0}}
{"type":"relationship","id":"r2","label":"EMPLOYS","start":{"id":"101"},"end":{"id":"102"},"properties":{"weight":0.9}}
"#;
        let (nodes, edges) = parse_apoc_jsonl(jsonl).expect("APOC JSONL should parse");
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
        assert_eq!(nodes[0].hierarchy_depth, 0); // Company root
        assert_eq!(nodes[1].hierarchy_depth, 1); // Department
        assert_eq!(nodes[2].hierarchy_depth, 2); // Engineer leaf
    }

    #[test]
    fn test_parse_cypher_script() {
        let cypher = r#"
CREATE (root:Concept {id: "c_root", title: "Mathematics"});
CREATE (sub:Concept {id: "c_geo", title: "Differential Geometry"});
CREATE (leaf:Concept {id: "c_lor", title: "Lorentz Manifold"});
CREATE (root)-[:SUBSUMES {weight: 1.0}]->(sub);
CREATE (sub)-[:SUBSUMES {weight: 0.95}]->(leaf);
"#;
        let (nodes, edges) = parse_cypher_script(cypher).expect("Cypher should parse");
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
        let root_node = nodes.iter().find(|n| n.external_id == "c_root").unwrap();
        let leaf_node = nodes.iter().find(|n| n.external_id == "c_lor").unwrap();
        assert_eq!(root_node.hierarchy_depth, 0);
        assert_eq!(leaf_node.hierarchy_depth, 2);
    }
}
