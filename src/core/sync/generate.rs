use crate::core::codec::{ValueEncoding, decode_delta, encode_delta};
use crate::core::packet::{EncodedDelta, SparseDelta};
use tracing::debug;

/// Generates a sparse delta from the difference between active and anchor weights.
///
/// Selects the top k parameters by absolute magnitude, where k is determined
/// by delta_selection_ratio, and encodes them. The anchor is updated with the
/// decoded values, exactly what receivers will apply, so anything lost to
/// selection or rounding stays in `active - anchor` and is sent later.
/// Non-finite differences are never selected, so a diverged local model
/// cannot spread NaN or infinity to its peers.
///
/// # Arguments
///
/// * `active` - Current model weights
/// * `anchor` - Reference weights (mutated in place)
/// * `delta_selection_ratio` - Fraction of parameters to include
/// * `seq` - Sequence number for ordering
/// * `encoding` - How values are encoded on the wire
///
/// # Returns
///
/// None if nothing survives selection and encoding, otherwise the delta as
/// receivers will decode it (indices ascending) and its wire form.
pub fn generate_local_delta(
    active: &[f32],
    anchor: &mut [f32],
    delta_selection_ratio: f32,
    seq: u64,
    encoding: ValueEncoding,
) -> Option<(SparseDelta, EncodedDelta)> {
    let mut deltas: Vec<(u32, f32)> = active
        .iter()
        .zip(anchor.iter())
        .enumerate()
        .map(|(i, (act, anc))| (i as u32, act - anc))
        .filter(|(_, d)| *d != 0.0 && d.is_finite())
        .collect();

    deltas.sort_unstable_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

    let k = (active.len() as f32 * delta_selection_ratio).ceil() as usize;
    deltas.truncate(k);
    deltas.sort_unstable_by_key(|&(idx, _)| idx);

    let selected = SparseDelta {
        sequence_id: seq,
        indices: deltas.iter().map(|&(idx, _)| idx).collect(),
        values: deltas.iter().map(|&(_, val)| val).collect(),
    };
    let encoded = encode_delta(&selected, encoding);
    let sent = decode_delta(&encoded).expect("own encoding must decode");

    if sent.indices.is_empty() {
        debug!("No significant deltas to share");
        return None;
    }

    for (&idx, &val) in sent.indices.iter().zip(&sent.values) {
        anchor[idx as usize] += val;
    }
    Some((sent, encoded))
}
