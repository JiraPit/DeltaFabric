#[cfg(test)]
mod integration_tests {
    use crate::core::packet::{FabricPacket, SparseDelta};
    use crate::core::sync::{apply_deltas, generate_local_delta, process_deltas};
    use std::collections::HashMap;

    #[test]
    fn test_end_to_end_delta_flow() {
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![0.0; 5];

        let delta = generate_local_delta(&active, &mut anchor, 0.4, 1, 1).unwrap();
        assert_eq!(delta.indices.len(), 2);

        let mut packet_updates: HashMap<u64, SparseDelta> = HashMap::new();
        packet_updates.insert(1, delta);

        let incoming = FabricPacket {
            updates: packet_updates,
        };

        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::new();

        let relay = process_deltas(&mut aggregator, &incoming, &mut seen_table, 1.0, 0.0);
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

        let fresh_packet = FabricPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 7,
                    indices: vec![0],
                    values: vec![1.0],
                },
            )]),
        };

        let stale_packet = FabricPacket {
            updates: HashMap::from([(
                2,
                SparseDelta {
                    sequence_id: 8,
                    indices: vec![1],
                    values: vec![2.0],
                },
            )]),
        };

        let relay_fresh = process_deltas(&mut aggregator, &fresh_packet, &mut seen_table, 1.0, 0.0);
        assert!(relay_fresh.is_some());
        assert_eq!(seen_table.get(&1), Some(&7));

        let relay_stale = process_deltas(&mut aggregator, &stale_packet, &mut seen_table, 1.0, 0.0);
        assert!(relay_stale.is_none());
        assert_eq!(seen_table.get(&2), Some(&10));
        assert_eq!(aggregator.len(), 1);
    }
}
