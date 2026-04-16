use crate::core::packet::SparseDelta;
use tracing::{debug, trace};

/// Generates a sparse delta from the difference between active and anchor weights.
///
/// Selects the top k parameters by absolute magnitude, where k is determined
/// by delta_selection_ratio. Updates anchor to match active for selected indices.
///
/// # Arguments
///
/// * `active` - Current model weights
/// * `anchor` - Reference weights (mutated in place)
/// * `delta_selection_ratio` - Fraction of parameters to include
/// * `_my_id` - Origin node ID (reserved for future use)
/// * `seq` - Sequence number for ordering
///
/// # Returns
///
/// None if no significant deltas, otherwise Some(SparseDelta).
pub fn generate_local_delta(
    active: &[f32],
    anchor: &mut [f32],
    delta_selection_ratio: f32,
    _my_id: u64,
    seq: u64,
) -> Option<SparseDelta> {
    let mut deltas: Vec<(u32, f32)> = active
        .iter()
        .zip(anchor.iter())
        .enumerate()
        .map(|(i, (act, anc))| (i as u32, act - anc))
        .collect();

    deltas.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    let k = (active.len() as f32 * delta_selection_ratio).ceil() as usize;
    let top_k = &deltas[..k.min(deltas.len())];

    let mut indices = Vec::new();
    let mut values = Vec::new();

    for &(idx, val) in top_k {
        if val != 0.0 {
            indices.push(idx);
            values.push(val);
            anchor[idx as usize] += val;
        }
    }

    if indices.is_empty() {
        debug!("No significant deltas to share");
        None
    } else {
        trace!(num_indices = %indices.len(), seq = %seq, "Generated local delta");
        Some(SparseDelta {
            sequence_id: seq,
            indices,
            values,
        })
    }
}
