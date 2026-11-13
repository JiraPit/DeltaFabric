use anyhow::{Context, Result};
use rkyv::{Archive, Deserialize, Serialize};

/// A packet carrying one node's compressed anchor change.
///
/// The origin node is identified by the topic it is published on, so it is
/// not repeated in the payload. Serialized with rkyv for efficient network
/// transmission.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct DeltaPacket {
    /// The compressed change applied to the sender's anchor
    pub delta: EncodedDelta,
    /// Hash of the sender's initial weights. Only set on its first packet,
    /// so receivers can detect nodes that started from different weights.
    pub init_hash: Option<u64>,
}

/// Wire form of a SparseDelta (see core::codec).
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct EncodedDelta {
    /// Sequence identifier, consecutive per origin starting at 1
    pub sequence_id: u64,
    /// Number of entries
    pub count: u32,
    /// Gaps between consecutive sorted indices (the first gap is the first
    /// index), each LEB128-encoded: about 1 byte per entry at 1% density
    pub index_gaps: Vec<u8>,
    /// The entry values
    pub values: EncodedValues,
}

/// Wire form of the delta values.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum EncodedValues {
    /// Exact values, 4 bytes each
    F32(Vec<f32>),
    /// value = data * scale, 1 byte each
    Int8 {
        /// Largest absolute value divided by 127
        scale: f32,
        /// Quantized values
        data: Vec<i8>,
    },
    /// value = +/- scale, 1 bit each
    Sign {
        /// Mean absolute value of the entries
        scale: f32,
        /// Bit i (LSB first) is set if entry i is negative
        negative: Vec<u8>,
    },
}

/// A sparse representation of parameter updates from a single node.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct SparseDelta {
    /// Sequence identifier, consecutive per origin starting at 1
    pub sequence_id: u64,
    /// Indices of parameters being updated
    pub indices: Vec<u32>,
    /// Delta values corresponding to each index
    pub values: Vec<f32>,
}

/// A full copy of a node's anchor, sent only when a peer needs to resync.
#[derive(Archive, Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct AnchorSnapshot {
    /// Sequence ID of the last delta included in this anchor (0 if none)
    pub sequence_id: u64,
    /// The sender's anchor weights
    pub anchor: Vec<f32>,
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

/// Serializes a value to rkyv bytes.
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn to_bytes<T>(value: &T) -> Result<Vec<u8>>
where
    T: for<'a> Serialize<
        rkyv::api::high::HighSerializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::rancor::Error,
        >,
    >,
{
    use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena};

    let mut arena = Arena::new();
    let bytes = to_bytes_with_alloc::<_, rkyv::rancor::Error>(value, arena.acquire())
        .context("rkyv serialization failed")?;
    Ok(bytes.to_vec())
}

/// Validates and deserializes a DeltaPacket from raw bytes.
///
/// # Errors
///
/// Returns an error if the bytes are not a valid archived DeltaPacket.
pub fn decode_packet(bytes: &[u8]) -> Result<DeltaPacket> {
    let archived = access_archived_packet(bytes).context("Invalid DeltaPacket bytes")?;
    rkyv::deserialize::<DeltaPacket, rkyv::rancor::Error>(archived)
        .context("Failed to deserialize DeltaPacket")
}

/// Validates and deserializes an AnchorSnapshot from raw bytes.
///
/// # Errors
///
/// Returns an error if the bytes are not a valid archived AnchorSnapshot.
pub fn decode_snapshot(bytes: &[u8]) -> Result<AnchorSnapshot> {
    let archived = rkyv::access::<ArchivedAnchorSnapshot, rkyv::rancor::Error>(bytes)
        .context("Invalid AnchorSnapshot bytes")?;
    rkyv::deserialize::<AnchorSnapshot, rkyv::rancor::Error>(archived)
        .context("Failed to deserialize AnchorSnapshot")
}

#[cfg(test)]
mod test;
