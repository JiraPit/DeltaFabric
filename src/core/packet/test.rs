#[cfg(test)]
mod packet_tests {
    use crate::core::codec::{ValueEncoding, encode_delta};
    use crate::core::packet::{
        AnchorSnapshot, DeltaPacket, SparseDelta, access_archived_packet, decode_packet,
        decode_snapshot, to_bytes,
    };
    use rkyv::deserialize;

    fn test_packet(init_hash: Option<u64>) -> DeltaPacket {
        let delta = SparseDelta {
            sequence_id: 42,
            indices: vec![0, 1, 2],
            values: vec![0.1, 0.2, 0.3],
        };
        DeltaPacket {
            delta: encode_delta(&delta, ValueEncoding::Int8),
            init_hash,
        }
    }

    #[test]
    fn test_fabric_packet_roundtrip() {
        let packet = test_packet(None);

        let bytes = to_bytes(&packet).unwrap();
        let archived = access_archived_packet(&bytes).unwrap();
        let deserialized: DeltaPacket =
            deserialize::<DeltaPacket, rkyv::rancor::Error>(archived).unwrap();

        assert_eq!(packet, deserialized);
    }

    #[test]
    fn test_fabric_packet_unaligned() {
        // With the 'unaligned' feature enabled in Cargo.toml, rkyv handles
        // unaligned buffers natively and safely in both debug and release.
        let packet = test_packet(Some(7));
        let bytes = to_bytes(&packet).unwrap();

        // Simulate unaligned network transmission by copying with offset
        let offset = 1;
        let mut unaligned = vec![0u8; bytes.len() + offset];
        unaligned[offset..].copy_from_slice(&bytes[..]);

        assert_eq!(decode_packet(&unaligned[offset..]).unwrap(), packet);
    }

    #[test]
    fn test_decode_packet_roundtrip_with_init_hash() {
        let packet = test_packet(Some(0xDEAD_BEEF));
        let bytes = to_bytes(&packet).unwrap();

        assert_eq!(decode_packet(&bytes).unwrap(), packet);
    }

    #[test]
    fn test_decode_packet_rejects_garbage() {
        assert!(decode_packet(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let snapshot = AnchorSnapshot {
            sequence_id: 9,
            anchor: vec![1.0, -2.5, 3.25],
        };
        let bytes = to_bytes(&snapshot).unwrap();

        assert_eq!(decode_snapshot(&bytes).unwrap(), snapshot);
    }

    #[test]
    fn test_packet_without_origin_is_smaller_than_snapshot() {
        // A delta packet must stay much smaller than a full anchor of the same model.
        let delta = SparseDelta {
            sequence_id: 1,
            indices: (0..10).collect(),
            values: vec![1.0; 10],
        };
        let packet = DeltaPacket {
            delta: encode_delta(&delta, ValueEncoding::F32),
            init_hash: None,
        };
        let snapshot = AnchorSnapshot {
            sequence_id: 1,
            anchor: vec![1.0; 1000],
        };

        let packet_len = to_bytes(&packet).unwrap().len();
        let snapshot_len = to_bytes(&snapshot).unwrap().len();
        assert!(packet_len < 10 * 8 + 64, "packet_len = {packet_len}");
        assert!(snapshot_len >= 1000 * 4);
    }
}
