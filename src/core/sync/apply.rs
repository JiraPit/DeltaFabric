use crate::core::packet::{ArchivedDeltaPacket, SparseDelta};
use std::collections::HashMap;
use tracing::{debug, trace};

pub fn process_deltas(
    aggregator: &mut HashMap<u32, f32>,
    incoming: &ArchivedDeltaPacket,
    seen_table: &mut HashMap<u64, u64>,
    alpha: f32,
    relay_threshold: f32,
) -> Option<HashMap<u64, SparseDelta>> {
    let mut relay_updates = HashMap::new();
    let mut fresh_count = 0;
    let mut stale_count = 0;

    for (origin_id, delta) in incoming.updates.iter() {
        let origin_id: u64 = (*origin_id).into();
        let last_seq = seen_table.get(&origin_id).copied().unwrap_or(0);
        let seq_id: u64 = delta.sequence_id.into();

        if seq_id > last_seq {
            fresh_count += 1;
            seen_table.insert(origin_id, seq_id);
            trace!(origin_id = %origin_id, seq = %seq_id, "Processing fresh delta");

            let mut relay_indices = Vec::new();
            let mut relay_values = Vec::new();

            for (i, &idx) in delta.indices.iter().enumerate() {
                let idx: u32 = idx.into();
                let val = delta.values[i];
                let damped_val = val * alpha;

                *aggregator.entry(idx).or_insert(0.0) += damped_val;

                if damped_val.abs() >= relay_threshold {
                    relay_indices.push(idx);
                    relay_values.push(damped_val);
                }
            }

            if !relay_indices.is_empty() {
                relay_updates.insert(
                    origin_id,
                    SparseDelta {
                        sequence_id: seq_id,
                        indices: relay_indices,
                        values: relay_values,
                    },
                );
            }
        } else {
            stale_count += 1;
            trace!(origin_id = %origin_id, seq = %seq_id, last_seq = %last_seq, "Skipping stale delta");
        }
    }

    if fresh_count > 0 || stale_count > 0 {
        debug!(
            fresh = %fresh_count,
            stale = %stale_count,
            relay_count = %relay_updates.len(),
            "Processed relay packet"
        );
    }

    if relay_updates.is_empty() {
        None
    } else {
        Some(relay_updates)
    }
}

pub fn apply_deltas(active: &mut [f32], anchor: &mut [f32], aggregator: &HashMap<u32, f32>) {
    for (&idx, &delta) in aggregator.iter() {
        active[idx as usize] += delta;
        anchor[idx as usize] += delta;
        trace!(idx = %idx, delta = %delta, "Applied delta to weights");
    }
}
