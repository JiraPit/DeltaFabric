use crate::core::packet::SparseDelta;
use std::collections::{HashMap, HashSet};

/// Weights a node gives itself and its neighbors when averaging anchors.
///
/// The weights of one node always sum to 1. Built with Metropolis weights,
/// the full matrix is symmetric, so mixing preserves the cluster average.
#[derive(Debug, Clone, PartialEq)]
pub struct MixingWeights {
    /// Weight of this node's own anchor
    pub self_weight: f32,
    /// Weight of each neighbor's anchor, keyed by node ID
    pub neighbors: HashMap<u64, f32>,
}

impl MixingWeights {
    /// Weights for a node with no neighbors (mixing has no effect).
    #[cfg(test)]
    pub fn isolated() -> Self {
        Self {
            self_weight: 1.0,
            neighbors: HashMap::new(),
        }
    }
}

/// Computes Metropolis mixing weights for a node from the undirected cluster graph.
///
/// For each neighbor j: w_ij = 1 / (1 + max(deg_i, deg_j)), and the node's
/// own weight is 1 minus the sum. Every node computes the same matrix from
/// the same graph, which keeps it symmetric.
///
/// # Arguments
///
/// * `my_id` - This node's ID
/// * `edges` - Undirected adjacency list of the cluster (self loops ignored)
///
/// # Returns
///
/// The weights this node uses for itself and each neighbor.
pub fn metropolis_weights(my_id: u64, edges: &HashMap<u64, HashSet<u64>>) -> MixingWeights {
    let degree = |id: u64| {
        edges
            .get(&id)
            .map_or(0, |n| n.iter().filter(|&&p| p != id).count())
    };

    let my_degree = degree(my_id);
    let mut neighbors = HashMap::new();
    if let Some(adjacent) = edges.get(&my_id) {
        for &peer in adjacent.iter().filter(|&&p| p != my_id) {
            let weight = 1.0 / (1 + my_degree.max(degree(peer))) as f32;
            neighbors.insert(peer, weight);
        }
    }

    let self_weight = 1.0 - neighbors.values().sum::<f32>();
    MixingWeights {
        self_weight,
        neighbors,
    }
}

/// Adds a weighted sparse delta into the mixing sum.
///
/// # Arguments
///
/// * `mix_sum` - Weighted sum of all anchors this node knows (mutated in place)
/// * `delta` - Sparse change to one node's anchor
/// * `weight` - Mixing weight of that node
pub fn accumulate(mix_sum: &mut [f32], delta: &SparseDelta, weight: f32) {
    for (&idx, &val) in delta.indices.iter().zip(delta.values.iter()) {
        if let Some(slot) = mix_sum.get_mut(idx as usize) {
            *slot += weight * val;
        }
    }
}

/// Moves the active weights toward the weighted average of all anchors.
///
/// Computes `active += gamma * (mix_sum - anchor)`. Since the weights sum to 1,
/// `mix_sum - anchor` equals the weighted sum of (neighbor anchor - own anchor),
/// so the pull is proportional to how far this node is from its neighbors.
///
/// # Arguments
///
/// * `active` - Current model weights (mutated in place)
/// * `anchor` - This node's anchor
/// * `mix_sum` - Weighted sum of all anchors this node knows
/// * `gamma` - Consensus step size
pub fn mix(active: &mut [f32], anchor: &[f32], mix_sum: &[f32], gamma: f32) {
    for ((act, &anc), &sum) in active.iter_mut().zip(anchor).zip(mix_sum) {
        *act += gamma * (sum - anc);
    }
}
