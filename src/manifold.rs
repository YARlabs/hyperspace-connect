//! Deterministic Manifold Embedding, Projection, and Collision-Resistant ID Mapping.
//!
//! Provides high-throughput vector projection routines for Zero-ETL ingestion when
//! migrating from graph databases (Neo4j/Neptune), message streams (Kafka/MQTT), or
//! legacy Euclidean vector stores (Qdrant/Pinecone/Milvus) into HyperspaceDB's
//! Lorentzian, Poincaré, or 801D Hybrid MRL manifolds.

use sha2::{Digest, Sha256};

/// Deterministically maps an arbitrary string or UUID identifier into a non-zero `u32` ID
/// using SHA-256 folding.
#[must_use]
pub fn hash_external_id_to_u32(external_id: &str) -> u32 {
    if let Ok(parsed) = external_id.trim().parse::<u32>() {
        if parsed > 0 {
            return parsed;
        }
    }
    let digest = Sha256::digest(external_id.as_bytes());
    let a = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let b = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    let folded = a ^ b.rotate_left(13);
    if folded == 0 {
        1
    } else {
        folded
    }
}

/// Target geometric manifold for Zero-ETL vector projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetManifold {
    /// Standard L2-normalized Cosine / Dot-product sphere.
    Cosine,
    /// Unnormalized Euclidean space.
    Euclidean,
    /// Lorentz hyperboloid ($\mathbb{H}^{d-1}$ in $\mathbb{R}^{1, d-1}$) with time-like coordinate $x_0 = \sqrt{1 + \|\mathbf{x}\|^2}$.
    Lorentz,
    /// Poincaré open unit ball ($\|\mathbf{x}\| < 1$).
    Poincare,
    /// HyperspaceDB 801D Hybrid MRL manifold (129D Lorentz prefix + 672D Cosine tail).
    Hybrid801,
}

impl TargetManifold {
    /// Parses a manifold name from string configuration.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "lorentz" | "hyperbolic" | "hyperboloid" => Self::Lorentz,
            "poincare" | "poincaré" | "ball" => Self::Poincare,
            "hybrid" | "hybrid801" | "801d" | "mrl801" => Self::Hybrid801,
            "l2" | "euclidean" => Self::Euclidean,
            _ => Self::Cosine,
        }
    }

    /// Returns the canonical HyperspaceDB metric string for collection creation.
    #[must_use]
    pub const fn metric_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "l2",
            Self::Lorentz => "lorentz",
            Self::Poincare => "poincare",
            Self::Hybrid801 => "hybrid",
        }
    }
}

/// Lifts an arbitrary spatial vector into the target manifold of the specified `dimension`.
///
/// - If `input` is shorter or longer than `dimension`, it is deterministically adapted
///   (truncated or frequency-folded) before manifold projection.
/// - For `TargetManifold::Lorentz`, index `0` is set to $x_0 = \sqrt{1 + \sum_{i=1}^{d-1} x_i^2}$
///   so that the Minkowski inner product satisfies $\langle \mathbf{x}, \mathbf{x} \rangle_L = -1$.
/// - For `TargetManifold::Poincare`, the vector is scaled to strictly reside inside the unit ball
///   ($\|\mathbf{x}\| \le 0.95$).
/// - For `TargetManifold::Hybrid801`, dimension is forced to `801` (or `dimension` if custom)
///   with a valid Lorentzian prefix at index `0..min(129, dimension)` and unit-normalized tail.
#[must_use]
pub fn project_vector_to_manifold(
    input: &[f64],
    dimension: usize,
    manifold: TargetManifold,
) -> Vec<f64> {
    let target_dim = if manifold == TargetManifold::Hybrid801 && dimension == 0 {
        801
    } else {
        dimension.max(2)
    };

    let mut out = vec![0.0_f64; target_dim];
    if !input.is_empty() {
        for (idx, &val) in input.iter().enumerate() {
            let clean = if val.is_finite() { val } else { 0.0 };
            out[idx % target_dim] += clean;
        }
    }

    match manifold {
        TargetManifold::Euclidean => out,
        TargetManifold::Cosine => {
            normalize_l2_in_place(&mut out);
            out
        }
        TargetManifold::Lorentz => {
            // Scale spatial components (1..target_dim) to moderate hyperbolic radius to prevent float overflow
            let spatial_norm_sq: f64 = out[1..].iter().map(|v| v * v).sum();
            if spatial_norm_sq > 25.0 {
                let scale = 5.0 / spatial_norm_sq.sqrt();
                for v in &mut out[1..] {
                    *v *= scale;
                }
            }
            let clamped_sq: f64 = out[1..].iter().map(|v| v * v).sum();
            out[0] = (1.0 + clamped_sq).sqrt();
            out
        }
        TargetManifold::Poincare => {
            let norm_sq: f64 = out.iter().map(|v| v * v).sum();
            if norm_sq > 0.0 {
                let norm = norm_sq.sqrt();
                // Map via tanh(norm) * 0.95 / norm so every point sits smoothly inside the open ball
                let scale = (norm.tanh() * 0.95) / norm;
                for v in &mut out {
                    *v *= scale;
                }
            }
            out
        }
        TargetManifold::Hybrid801 => {
            let lorentz_cutoff = target_dim.min(129);
            // 1. Project prefix [0..lorentz_cutoff] onto the Lorentz hyperboloid
            let spatial_norm_sq: f64 = out[1..lorentz_cutoff].iter().map(|v| v * v).sum();
            if spatial_norm_sq > 16.0 {
                let scale = 4.0 / spatial_norm_sq.sqrt();
                for v in &mut out[1..lorentz_cutoff] {
                    *v *= scale;
                }
            }
            let prefix_sq: f64 = out[1..lorentz_cutoff].iter().map(|v| v * v).sum();
            out[0] = (1.0 + prefix_sq).sqrt();

            // 2. Unit-normalize the Euclidean/Cosine tail [lorentz_cutoff..target_dim]
            if target_dim > lorentz_cutoff {
                normalize_l2_in_place(&mut out[lorentz_cutoff..]);
            }
            out
        }
    }
}

/// Deterministically embeds structured text and hierarchy depth into the requested manifold.
///
/// Uses multi-gram cryptographic feature hashing so that semantically overlapping labels
/// and properties produce correlated coordinates without requiring an external GPU service.
#[must_use]
pub fn embed_text_deterministic(
    text: &str,
    hierarchy_depth: u32,
    dimension: usize,
    manifold: TargetManifold,
) -> Vec<f64> {
    let target_dim = dimension.max(4);
    let mut raw = vec![0.0_f64; target_dim];

    let normalized = text.trim().to_ascii_lowercase();
    for (pos, token) in normalized
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .enumerate()
    {
        let digest = Sha256::digest(token.as_bytes());
        let pos_weight = 1.0 / (1.0 + (pos as f64) * 0.02);
        for chunk in digest.chunks_exact(4) {
            let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let slot = (h as usize) % target_dim;
            // Ensure slot 0 is not overwritten before Lorentz lifting
            let effective_slot = if slot == 0 && target_dim > 1 { 1 } else { slot };
            let sign = if (h & 0x8000_0000) != 0 { 1.0 } else { -1.0 };
            raw[effective_slot] += sign * pos_weight;
        }
    }

    // Encode hierarchy depth into radial scale for hyperbolic manifolds
    if matches!(manifold, TargetManifold::Lorentz | TargetManifold::Hybrid801) {
        let radial_boost = 0.25 + f64::from(hierarchy_depth.min(16)) * 0.35;
        let current_norm_sq: f64 = raw[1..].iter().map(|v| v * v).sum();
        if current_norm_sq > 1e-12 {
            let scale = radial_boost / current_norm_sq.sqrt();
            for v in &mut raw[1..] {
                *v *= scale;
            }
        } else if target_dim > 1 {
            raw[1] = radial_boost;
        }
    }

    project_vector_to_manifold(&raw, target_dim, manifold)
}

fn normalize_l2_in_place(slice: &mut [f64]) {
    let norm_sq: f64 = slice.iter().map(|v| v * v).sum();
    if norm_sq > 1e-12 {
        let inv = 1.0 / norm_sq.sqrt();
        for v in slice.iter_mut() {
            *v *= inv;
        }
    } else if !slice.is_empty() {
        slice[0] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_projection_invariant() {
        let input = vec![0.5, -1.2, 2.3, 0.8, -0.4];
        let projected = project_vector_to_manifold(&input, 8, TargetManifold::Lorentz);
        assert_eq!(projected.len(), 8);
        // Minkowski inner product: -x0^2 + sum(x_i^2) == -1.0
        let spatial_sq: f64 = projected[1..].iter().map(|v| v * v).sum();
        let minkowski = -projected[0] * projected[0] + spatial_sq;
        assert!((minkowski - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_hybrid_801_projection() {
        let vec = embed_text_deterministic("Person:Alice knows Bob", 2, 801, TargetManifold::Hybrid801);
        assert_eq!(vec.len(), 801);
        let prefix_sq: f64 = vec[1..129].iter().map(|v| v * v).sum();
        let minkowski = -vec[0] * vec[0] + prefix_sq;
        assert!((minkowski - (-1.0)).abs() < 1e-9);

        let tail_norm_sq: f64 = vec[129..].iter().map(|v| v * v).sum();
        assert!((tail_norm_sq - 1.0).abs() < 1e-9);
    }
}

