use crate::core::packet::SparseDelta;
use tracing::{debug, trace};

pub fn generate_local_delta(
    active: &[f32],
    anchor: &mut [f32],
    top_k_pct: f32,
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

    let k = (active.len() as f32 * top_k_pct).ceil() as usize;
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
