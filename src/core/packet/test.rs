#[cfg(test)]
mod packet_tests {
    use crate::core::{DeltaPacket, SparseDelta};
    use rkyv::{api::high::to_bytes_with_alloc, deserialize, ser::allocator::Arena};
    use std::collections::HashMap;

    #[test]
    fn test_fabric_packet_roundtrip() {
        let packet = DeltaPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 42,
                    indices: vec![0, 1, 2],
                    values: vec![0.1, 0.2, 0.3],
                },
            )]),
        };

        let mut arena = Arena::new();
        let bytes =
            to_bytes_with_alloc::<_, rkyv::rancor::Error>(&packet, arena.acquire()).unwrap();

        let archived =
            rkyv::access::<rkyv::Archived<DeltaPacket>, rkyv::rancor::Error>(&bytes[..]).unwrap();
        let deserialized: DeltaPacket =
            deserialize::<DeltaPacket, rkyv::rancor::Error>(archived).unwrap();

        assert_eq!(packet, deserialized);
    }

    #[test]
    fn test_sparse_delta_empty() {
        let delta = SparseDelta {
            sequence_id: 0,
            indices: vec![],
            values: vec![],
        };

        assert!(delta.indices.is_empty());
        assert!(delta.values.is_empty());
    }
}
