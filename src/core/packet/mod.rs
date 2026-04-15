use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct DeltaPacket {
    pub updates: HashMap<u64, SparseDelta>,
}

#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct SparseDelta {
    pub sequence_id: u64,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

pub fn access_archived_packet<'a>(
    bytes: &'a [u8],
) -> Result<&'a ArchivedDeltaPacket, rkyv::rancor::Error> {
    rkyv::access::<ArchivedDeltaPacket, rkyv::rancor::Error>(bytes)
}

#[cfg(test)]
mod test;
