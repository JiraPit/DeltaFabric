#[cfg(test)]
mod integration_tests {
    use crate::core::packet::{DeltaPacket, SparseDelta};
    use crate::core::sync::{apply_deltas, generate_local_delta, process_deltas};
    use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena};
    use std::collections::HashMap;

    #[test]
    fn test_end_to_end_delta_flow() {
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![0.0; 5];

        let delta = generate_local_delta(&active, &mut anchor, 0.4, 1, 1).unwrap();
        assert_eq!(delta.indices.len(), 2);

        let packet = DeltaPacket {
            updates: HashMap::from([(1, delta)]),
        };

        let bytes =
            to_bytes_with_alloc::<_, rkyv::rancor::Error>(&packet, Arena::new().acquire()).unwrap();
        let archived = crate::core::access_archived_packet(&bytes).unwrap();

        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::new();

        let relay = process_deltas(&mut aggregator, archived, &mut seen_table, 1.0, 0.0);
        assert!(relay.is_some());

        let mut active_weights = active.clone();
        apply_deltas(&mut active_weights, &mut anchor, &aggregator);

        assert_eq!(active_weights[4], 10.0);
        assert_eq!(active_weights[3], 8.0);
    }

    #[test]
    fn test_seen_table_deduplication() {
        let mut seen_table: HashMap<u64, u64> = HashMap::from([(1, 5), (2, 10)]);
        let mut aggregator: HashMap<u32, f32> = HashMap::new();

        let fresh_packet = DeltaPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 7,
                    indices: vec![0],
                    values: vec![1.0],
                },
            )]),
        };

        let stale_packet = DeltaPacket {
            updates: HashMap::from([(
                2,
                SparseDelta {
                    sequence_id: 8,
                    indices: vec![1],
                    values: vec![2.0],
                },
            )]),
        };

        let bytes_fresh =
            to_bytes_with_alloc::<_, rkyv::rancor::Error>(&fresh_packet, Arena::new().acquire())
                .unwrap();
        let archived_fresh = crate::core::access_archived_packet(&bytes_fresh).unwrap();

        let relay_fresh =
            process_deltas(&mut aggregator, archived_fresh, &mut seen_table, 1.0, 0.0);
        assert!(relay_fresh.is_some());
        assert_eq!(seen_table.get(&1), Some(&7));

        let bytes_stale =
            to_bytes_with_alloc::<_, rkyv::rancor::Error>(&stale_packet, Arena::new().acquire())
                .unwrap();
        let archived_stale = crate::core::access_archived_packet(&bytes_stale).unwrap();

        let relay_stale =
            process_deltas(&mut aggregator, archived_stale, &mut seen_table, 1.0, 0.0);
        assert!(relay_stale.is_none());
        assert_eq!(seen_table.get(&2), Some(&10));
        assert_eq!(aggregator.len(), 1);
    }
}
