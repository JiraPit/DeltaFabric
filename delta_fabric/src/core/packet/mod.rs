use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct FabricPacket {
    pub updates: HashMap<u64, SparseDelta>,
}

#[derive(Archive, Serialize, Deserialize, PartialEq, Debug)]
pub struct SparseDelta {
    pub sequence_id: u64,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

pub type ArchivedFabricPacketType = <FabricPacket as Archive>::Archived;

#[cfg(test)]
mod test;
