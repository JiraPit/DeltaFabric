use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// A packet containing parameter deltas from one or more nodes.
///
/// Serialized with rkyv for efficient network transmission.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct DeltaPacket {
    /// Maps origin node ID to their sparse delta update
    pub updates: HashMap<u64, SparseDelta>,
}

/// A sparse representation of parameter updates from a single node.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct SparseDelta {
    /// Sequence identifier for ordering and deduplication
    pub sequence_id: u64,
    /// Indices of parameters being updated
    pub indices: Vec<u32>,
    /// Delta values corresponding to each index
    pub values: Vec<f32>,
}

/// Accesses an archived DeltaPacket from raw bytes without deserializing.
///
/// # Arguments
///
/// * `bytes` - Serialized bytes of an archived DeltaPacket
///
/// # Returns
///
/// Reference to the archived packet, or an error if access fails.
pub fn access_archived_packet(bytes: &[u8]) -> Result<&ArchivedDeltaPacket, rkyv::rancor::Error> {
    rkyv::access::<ArchivedDeltaPacket, rkyv::rancor::Error>(bytes)
}

#[cfg(test)]
mod test;
