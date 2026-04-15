#[cfg(test)]
mod sync_tests {
    use crate::core::packet::{FabricPacket, SparseDelta};
    use crate::core::sync::{apply_deltas, generate_local_delta, process_deltas};
    use std::collections::HashMap;

    #[test]
    fn test_generate_local_delta_empty() {
        let active = vec![0.0, 0.0, 0.0];
        let mut anchor = vec![0.0, 0.0, 0.0];

        let result = generate_local_delta(&active, &mut anchor, 0.1, 1, 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_generate_local_delta_top_k() {
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![0.0; 5];

        let result = generate_local_delta(&active, &mut anchor, 0.4, 1, 1).unwrap();

        assert_eq!(result.indices.len(), 2);
        assert_eq!(result.indices, vec![4, 3]);
        assert_eq!(result.values, vec![5.0, 4.0]);

        assert_eq!(anchor[4], 5.0);
        assert_eq!(anchor[3], 4.0);
    }

    #[test]
    fn test_apply_deltas() {
        let mut active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let aggregator: HashMap<u32, f32> = HashMap::from([(0, 0.5), (2, 1.0)]);

        apply_deltas(&mut active, &mut anchor, &aggregator);

        assert_eq!(active[0], 1.5);
        assert_eq!(active[1], 2.0);
        assert_eq!(active[2], 4.0);
        assert_eq!(anchor[0], 1.5);
        assert_eq!(anchor[2], 4.0);
    }

    #[test]
    fn test_apply_deltas_empty_aggregator() {
        let mut active = vec![1.0, 2.0, 3.0];
        let mut anchor = vec![1.0, 2.0, 3.0];
        let aggregator: HashMap<u32, f32> = HashMap::new();

        apply_deltas(&mut active, &mut anchor, &aggregator);

        assert_eq!(active, vec![1.0, 2.0, 3.0]);
        assert_eq!(anchor, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_process_deltas_fresh_delta() {
        let incoming = FabricPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 10,
                    indices: vec![0, 1],
                    values: vec![0.5, 1.0],
                },
            )]),
        };
        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::new();

        let relay = process_deltas(&mut aggregator, &incoming, &mut seen_table, 1.0, 0.0);

        assert!(relay.is_some());
        assert_eq!(seen_table.get(&1), Some(&10));
        assert_eq!(aggregator.get(&0), Some(&0.5));
        assert_eq!(aggregator.get(&1), Some(&1.0));
    }

    #[test]
    fn test_process_deltas_stale_delta() {
        let incoming = FabricPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 5,
                    indices: vec![0],
                    values: vec![1.0],
                },
            )]),
        };
        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::from([(1, 10)]);

        let relay = process_deltas(&mut aggregator, &incoming, &mut seen_table, 1.0, 0.0);

        assert!(relay.is_none());
        assert!(aggregator.is_empty());
        assert_eq!(seen_table.get(&1), Some(&10));
    }

    #[test]
    fn test_process_deltas_threshold() {
        let incoming = FabricPacket {
            updates: HashMap::from([(
                1,
                SparseDelta {
                    sequence_id: 1,
                    indices: vec![0, 1, 2],
                    values: vec![0.1, 1.0, 0.05],
                },
            )]),
        };
        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::new();

        let relay = process_deltas(&mut aggregator, &incoming, &mut seen_table, 1.0, 0.5);

        assert!(relay.is_some());
        let relay_updates = relay.unwrap();
        assert!(relay_updates.contains_key(&1));
        let relayed = &relay_updates[&1];
        assert_eq!(relayed.indices, vec![1]);
        assert_eq!(relayed.values, vec![1.0]);
    }
}
