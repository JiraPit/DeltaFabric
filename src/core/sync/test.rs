#[cfg(test)]
mod generate_tests {
    use crate::core::codec::{ValueEncoding, decode_delta};
    use crate::core::sync::generate::generate_local_delta;

    const ALL: [ValueEncoding; 3] = [ValueEncoding::F32, ValueEncoding::Int8, ValueEncoding::Sign];

    /// Deterministic pseudo-random values in [-1, 1].
    fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_generate_local_delta_empty() {
        let active = vec![0.0; 5];
        let mut anchor = vec![0.0; 5];

        for encoding in ALL {
            assert!(generate_local_delta(&active, &mut anchor, 0.1, 1, encoding).is_none());
        }
    }

    #[test]
    fn test_generate_local_delta_top_k() {
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![0.0; 5];

        let (sent, encoded) =
            generate_local_delta(&active, &mut anchor, 0.4, 1, ValueEncoding::F32).unwrap();

        assert_eq!(sent.sequence_id, 1);
        assert_eq!(encoded.sequence_id, 1);
        assert_eq!(
            sent.indices,
            vec![3, 4],
            "indices are sent in ascending order"
        );
        assert_eq!(sent.values, vec![4.0, 5.0]);
        assert_eq!(anchor, vec![0.0, 0.0, 0.0, 4.0, 5.0]);
    }

    #[test]
    fn test_generate_local_delta_selects_by_absolute_value() {
        let active = vec![0.1, -3.0, 2.0];
        let mut anchor = vec![0.0; 3];

        let (sent, _) =
            generate_local_delta(&active, &mut anchor, 0.33, 1, ValueEncoding::F32).unwrap();

        assert_eq!(sent.indices, vec![1]);
        assert_eq!(sent.values, vec![-3.0]);
        assert_eq!(anchor[1], -3.0);
    }

    #[test]
    fn test_generate_local_delta_residual_sent_later() {
        // Error feedback: what is not selected now stays in active - anchor.
        let active = vec![1.0, 2.0, 3.0, 4.0];
        let mut anchor = vec![0.0; 4];
        let f32 = ValueEncoding::F32;

        let (first, _) = generate_local_delta(&active, &mut anchor, 0.5, 1, f32).unwrap();
        let (second, _) = generate_local_delta(&active, &mut anchor, 0.5, 2, f32).unwrap();
        let third = generate_local_delta(&active, &mut anchor, 0.5, 3, f32);

        assert_eq!(first.indices, vec![2, 3]);
        assert_eq!(second.indices, vec![0, 1]);
        assert!(third.is_none());
        assert_eq!(anchor, active);
    }

    #[test]
    fn test_generate_local_delta_skips_non_finite() {
        let active = vec![f32::NAN, 1.0, f32::INFINITY, 0.5];

        for encoding in ALL {
            let mut anchor = vec![0.0; 4];
            let (sent, _) = generate_local_delta(&active, &mut anchor, 1.0, 1, encoding).unwrap();

            assert_eq!(sent.indices, vec![1, 3]);
            assert!(sent.values.iter().all(|v| v.is_finite()));
            assert_eq!(anchor[0], 0.0);
            assert_eq!(anchor[2], 0.0);
        }
    }

    #[test]
    fn test_anchor_moves_by_exactly_what_receivers_decode() {
        // The safety condition for lossy encodings: the anchor must record the
        // rounded values that were sent, not the original ones.
        let active = pseudo_random(5, 300);
        for encoding in ALL {
            let mut anchor = vec![0.0; 300];
            let before = anchor.clone();
            let (sent, encoded) =
                generate_local_delta(&active, &mut anchor, 0.1, 1, encoding).unwrap();

            let decoded = decode_delta(&encoded).unwrap();
            assert_eq!(decoded, sent, "{encoding:?}");
            let mut expected = before;
            for (&i, &v) in decoded.indices.iter().zip(&decoded.values) {
                expected[i as usize] += v;
            }
            assert_eq!(anchor, expected, "{encoding:?}");
        }
    }

    #[test]
    fn test_lossy_encodings_eventually_send_everything() {
        // Repeated syncs without training drive active - anchor to zero.
        let active = pseudo_random(9, 200);
        for encoding in ALL {
            let mut anchor = vec![0.0; 200];
            for seq in 1..=400 {
                if generate_local_delta(&active, &mut anchor, 0.1, seq, encoding).is_none() {
                    break;
                }
            }
            let max_left = active
                .iter()
                .zip(&anchor)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_left < 1e-3, "{encoding:?}: {max_left}");
        }
    }
}

#[cfg(test)]
mod mixing_tests {
    use crate::core::packet::SparseDelta;
    use crate::core::sync::mixing::{MixingWeights, accumulate, metropolis_weights, mix};
    use std::collections::{HashMap, HashSet};

    fn graph(links: &[(u64, u64)]) -> HashMap<u64, HashSet<u64>> {
        let mut edges: HashMap<u64, HashSet<u64>> = HashMap::new();
        for &(a, b) in links {
            edges.entry(a).or_default().insert(b);
            edges.entry(b).or_default().insert(a);
        }
        edges
    }

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    /// Checks that every row sums to 1 and the matrix is symmetric.
    fn assert_doubly_stochastic(edges: &HashMap<u64, HashSet<u64>>) {
        let weights: HashMap<u64, MixingWeights> = edges
            .keys()
            .map(|&id| (id, metropolis_weights(id, edges)))
            .collect();
        for (id, w) in &weights {
            assert_close(w.self_weight + w.neighbors.values().sum::<f32>(), 1.0);
            assert!(w.self_weight >= 0.0);
            for (peer, weight) in &w.neighbors {
                assert_close(*weight, weights[peer].neighbors[id]);
            }
        }
    }

    #[test]
    fn test_metropolis_full_mesh_is_plain_average() {
        let edges = graph(&[(1, 2), (1, 3), (2, 3)]);
        let w = metropolis_weights(1, &edges);

        assert_close(w.self_weight, 1.0 / 3.0);
        assert_close(w.neighbors[&2], 1.0 / 3.0);
        assert_close(w.neighbors[&3], 1.0 / 3.0);
    }

    #[test]
    fn test_metropolis_two_nodes() {
        let edges = graph(&[(1, 2)]);
        let w = metropolis_weights(2, &edges);

        assert_close(w.self_weight, 0.5);
        assert_close(w.neighbors[&1], 0.5);
    }

    #[test]
    fn test_metropolis_star_uses_larger_degree() {
        let edges = graph(&[(1, 2), (1, 3), (1, 4)]);
        let center = metropolis_weights(1, &edges);
        let leaf = metropolis_weights(2, &edges);

        assert_close(center.neighbors[&2], 0.25);
        assert_close(center.self_weight, 0.25);
        assert_close(leaf.neighbors[&1], 0.25);
        assert_close(leaf.self_weight, 0.75);
        assert_doubly_stochastic(&edges);
    }

    #[test]
    fn test_metropolis_irregular_graph_is_doubly_stochastic() {
        let edges = graph(&[(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (4, 5), (5, 6)]);
        assert_doubly_stochastic(&edges);
    }

    #[test]
    fn test_metropolis_isolated_and_self_loops() {
        let isolated = metropolis_weights(7, &HashMap::new());
        assert_eq!(isolated, MixingWeights::isolated());

        let edges = graph(&[(1, 1), (1, 2)]);
        let w = metropolis_weights(1, &edges);
        assert!(!w.neighbors.contains_key(&1));
        assert_close(w.neighbors[&2], 0.5);
        assert_close(w.self_weight, 0.5);
    }

    #[test]
    fn test_accumulate_applies_weight() {
        let mut mix_sum = vec![1.0, 1.0, 1.0];
        let delta = SparseDelta {
            sequence_id: 1,
            indices: vec![0, 2, 99],
            values: vec![2.0, -4.0, 5.0],
        };

        accumulate(&mut mix_sum, &delta, 0.5);

        assert_eq!(mix_sum, vec![2.0, 1.0, -1.0]);
    }

    #[test]
    fn test_mix_moves_toward_mix_sum() {
        let mut active = vec![1.0, 0.0];
        let anchor = vec![1.0, 2.0];
        let mix_sum = vec![3.0, 2.0];

        mix(&mut active, &anchor, &mix_sum, 0.5);

        assert_eq!(active, vec![2.0, 0.0]);
    }
}

#[cfg(test)]
mod state_tests {
    use crate::core::codec::{ValueEncoding, decode_delta, encode_delta};
    use crate::core::packet::{AnchorSnapshot, DeltaPacket, SparseDelta};
    use crate::core::sync::MixingWeights;
    use crate::core::sync::state::{ReceiveOutcome, SyncState, weights_hash};
    use std::collections::HashMap;

    /// Node with two neighbors (2 and 3), equal weights of 1/3.
    fn mesh_state(gamma: f32) -> SyncState {
        let weights = MixingWeights {
            self_weight: 1.0 / 3.0,
            neighbors: HashMap::from([(2, 1.0 / 3.0), (3, 1.0 / 3.0)]),
        };
        SyncState::new(gamma, 1.0, weights)
    }

    fn packet(seq: u64, indices: Vec<u32>, values: Vec<f32>) -> DeltaPacket {
        let delta = SparseDelta {
            sequence_id: seq,
            indices,
            values,
        };
        DeltaPacket {
            delta: encode_delta(&delta, ValueEncoding::F32),
            init_hash: None,
        }
    }

    fn assert_all_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn test_initialize_sets_anchor_and_mix_sum() {
        let mut state = mesh_state(0.5);
        assert!(!state.is_initialized());

        state.initialize(&[1.0, 2.0]);

        assert!(state.is_initialized());
        assert_eq!(state.anchor(), &[1.0, 2.0]);
        assert_eq!(state.mix_sum(), &[1.0, 2.0]);
        assert_eq!(state.local_sequence(), 0);
    }

    #[test]
    fn test_receive_before_initialize_is_ignored() {
        let mut state = mesh_state(0.5);
        let outcome = state.receive(2, packet(1, vec![0], vec![1.0]));
        assert_eq!(outcome, ReceiveOutcome::NotInitialized);
    }

    #[test]
    fn test_receive_updates_only_mix_sum() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0, 0.0]);

        let outcome = state.receive(2, packet(1, vec![1], vec![3.0]));

        assert_eq!(outcome, ReceiveOutcome::Applied);
        assert_eq!(state.anchor(), &[0.0, 0.0]);
        assert_all_close(state.mix_sum(), &[0.0, 1.0]);
    }

    #[test]
    fn test_receive_duplicate_is_ignored() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);

        state.receive(2, packet(1, vec![0], vec![3.0]));
        let outcome = state.receive(2, packet(1, vec![0], vec![3.0]));

        assert_eq!(outcome, ReceiveOutcome::Duplicate);
        assert_all_close(state.mix_sum(), &[1.0]);
    }

    #[test]
    fn test_receive_malformed_is_ignored_without_advancing() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);

        let mut broken = packet(1, vec![0], vec![3.0]);
        broken.delta.count = 5;
        assert_eq!(state.receive(2, broken), ReceiveOutcome::Malformed);
        assert_eq!(state.mix_sum(), &[0.0]);

        // Seq 1 is still expected, so a valid copy is applied.
        assert_eq!(
            state.receive(2, packet(1, vec![0], vec![3.0])),
            ReceiveOutcome::Applied
        );
    }

    #[test]
    fn test_sync_step_uses_configured_encoding() {
        for encoding in [ValueEncoding::Int8, ValueEncoding::Sign] {
            let mut state = mesh_state(0.0).with_encoding(encoding);
            state.initialize(&[0.0; 4]);

            let mut active = vec![1.0, -0.3, 0.2, 2.0];
            let sent = state.sync_step(&mut active).unwrap().unwrap();
            let decoded = decode_delta(&sent.delta).unwrap();

            // The anchor holds exactly the decoded (rounded) values.
            let mut expected = vec![0.0; 4];
            for (&i, &v) in decoded.indices.iter().zip(&decoded.values) {
                expected[i as usize] = v;
            }
            assert_eq!(state.anchor(), expected.as_slice(), "{encoding:?}");
            assert_ne!(state.anchor(), active.as_slice(), "{encoding:?} is lossy");
        }
    }

    #[test]
    fn test_receive_unknown_peer_is_ignored() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);

        let outcome = state.receive(42, packet(1, vec![0], vec![3.0]));

        assert_eq!(outcome, ReceiveOutcome::UnknownPeer);
        assert_eq!(state.mix_sum(), &[0.0]);
    }

    #[test]
    fn test_receive_gap_starts_resync_and_buffers() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);

        let outcome = state.receive(2, packet(2, vec![0], vec![3.0]));

        assert_eq!(outcome, ReceiveOutcome::ResyncStarted);
        assert!(state.is_resyncing());
        assert_eq!(state.missing_snapshots(), vec![2, 3]);
        assert_eq!(
            state.receive(3, packet(1, vec![0], vec![1.0])),
            ReceiveOutcome::Buffered
        );
        assert_eq!(state.mix_sum(), &[0.0]);
    }

    #[test]
    fn test_receive_checks_init_hash_on_first_packet() {
        let mut state = mesh_state(0.5);
        state.initialize(&[1.0, 2.0]);
        let mine = weights_hash(&[1.0, 2.0]);

        let mut matching = packet(1, vec![0], vec![1.0]);
        matching.init_hash = Some(mine);
        assert_eq!(state.receive(2, matching), ReceiveOutcome::Applied);

        let mut different = packet(1, vec![0], vec![1.0]);
        different.init_hash = Some(mine ^ 1);
        assert_eq!(state.receive(3, different), ReceiveOutcome::ResyncStarted);
    }

    #[test]
    fn test_sync_step_sequence_only_advances_when_sending() {
        let mut state = mesh_state(0.0);
        state.initialize(&[0.0, 0.0]);

        let mut unchanged = vec![0.0, 0.0];
        assert!(state.sync_step(&mut unchanged).unwrap().is_none());
        assert_eq!(state.local_sequence(), 0);

        let mut changed = vec![1.0, 0.0];
        let first = state.sync_step(&mut changed).unwrap().unwrap();
        assert_eq!(first.delta.sequence_id, 1);
        assert_eq!(first.init_hash, Some(weights_hash(&[0.0, 0.0])));

        changed[1] = 2.0;
        let second = state.sync_step(&mut changed).unwrap().unwrap();
        assert_eq!(second.delta.sequence_id, 2);
        assert_eq!(second.init_hash, None);
        assert_eq!(state.local_sequence(), 2);
    }

    #[test]
    fn test_sync_step_adds_own_delta_to_mix_sum() {
        let mut state = mesh_state(0.0);
        state.initialize(&[0.0]);

        let mut active = vec![3.0];
        state.sync_step(&mut active).unwrap();

        assert_eq!(state.anchor(), &[3.0]);
        assert_all_close(state.mix_sum(), &[1.0]);
    }

    #[test]
    fn test_sync_step_gamma_zero_leaves_active_unchanged() {
        let mut state = mesh_state(0.0);
        state.initialize(&[0.0, 0.0]);
        state.receive(2, packet(1, vec![0, 1], vec![9.0, -9.0]));

        let mut active = vec![1.0, 1.0];
        state.sync_step(&mut active).unwrap();

        assert_eq!(active, vec![1.0, 1.0]);
    }

    #[test]
    fn test_sync_step_mixes_toward_neighbors() {
        // Two nodes, weights 1/2. Neighbor's anchor moved to 4, ours to 2.
        let weights = MixingWeights {
            self_weight: 0.5,
            neighbors: HashMap::from([(2, 0.5)]),
        };
        let mut state = SyncState::new(0.5, 1.0, weights);
        state.initialize(&[0.0]);
        state.receive(2, packet(1, vec![0], vec![4.0]));

        let mut active = vec![2.0];
        state.sync_step(&mut active).unwrap();

        // Mix first: mix_sum = 0.5 * 0 + 0.5 * 4 = 2, active = 2 + 0.5 * (2 - 0) = 3.
        // Then send 3: anchor = 3, mix_sum = 2 + 0.5 * 3 = 3.5.
        assert_all_close(&active, &[3.0]);
        assert_eq!(state.anchor(), &[3.0]);
        assert_all_close(state.mix_sum(), &[3.5]);
    }

    #[test]
    fn test_sync_step_pauses_mixing_during_resync() {
        let mut state = mesh_state(1.0);
        state.initialize(&[0.0]);
        state.receive(2, packet(5, vec![0], vec![9.0]));
        assert!(state.is_resyncing());

        let mut active = vec![3.0];
        let sent = state.sync_step(&mut active).unwrap();

        assert!(sent.is_some(), "own deltas keep flowing during a resync");
        assert_eq!(active, vec![3.0]);
    }

    #[test]
    fn test_sync_step_errors() {
        let mut state = mesh_state(0.5);
        assert!(state.sync_step(&mut [1.0]).is_err());

        state.initialize(&[0.0, 0.0]);
        assert!(state.sync_step(&mut [1.0]).is_err());
    }

    #[test]
    fn test_snapshot_reports_anchor_and_sequence() {
        let mut state = mesh_state(0.0);
        state.initialize(&[0.0]);
        state.sync_step(&mut [2.0]).unwrap();

        assert_eq!(
            state.snapshot(),
            AnchorSnapshot {
                sequence_id: 1,
                anchor: vec![2.0],
            }
        );
    }

    #[test]
    fn test_receive_snapshot_rejections() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);
        let snapshot = AnchorSnapshot {
            sequence_id: 0,
            anchor: vec![0.0],
        };

        assert!(state.receive_snapshot(2, snapshot.clone()).is_err());

        state.receive(2, packet(3, vec![0], vec![1.0]));
        assert!(state.receive_snapshot(42, snapshot.clone()).is_err());
        let wrong_len = AnchorSnapshot {
            sequence_id: 0,
            anchor: vec![0.0, 0.0],
        };
        assert!(state.receive_snapshot(2, wrong_len).is_err());
        assert!(state.is_resyncing());
    }

    #[test]
    fn test_resync_rebuilds_mix_sum_and_replays_newer_deltas() {
        let mut state = mesh_state(0.5);
        state.initialize(&[0.0]);
        state.sync_step(&mut [3.0]).unwrap();

        // Peer 2's seq 1 was lost; seq 2 arrives and starts a resync.
        assert_eq!(
            state.receive(2, packet(2, vec![0], vec![1.0])),
            ReceiveOutcome::ResyncStarted
        );
        // Peer 3's seq 1 and 2 arrive while waiting.
        state.receive(3, packet(1, vec![0], vec![2.0]));
        state.receive(3, packet(2, vec![0], vec![4.0]));

        // Peer 2's snapshot already includes its seq 2; peer 3's includes only seq 1.
        let done = state
            .receive_snapshot(
                2,
                AnchorSnapshot {
                    sequence_id: 2,
                    anchor: vec![6.0],
                },
            )
            .unwrap();
        assert!(!done);
        assert_eq!(state.missing_snapshots(), vec![3]);

        let done = state
            .receive_snapshot(
                3,
                AnchorSnapshot {
                    sequence_id: 1,
                    anchor: vec![2.0],
                },
            )
            .unwrap();
        assert!(done);
        assert!(!state.is_resyncing());

        // Peer 3's anchor after replaying seq 2 is 6: (3 + 6 + 6) / 3 = 5.
        assert_all_close(state.mix_sum(), &[5.0]);
        assert_eq!(
            state.receive(2, packet(2, vec![0], vec![1.0])),
            ReceiveOutcome::Duplicate
        );
        assert_eq!(
            state.receive(3, packet(3, vec![0], vec![3.0])),
            ReceiveOutcome::Applied
        );
        assert_all_close(state.mix_sum(), &[6.0]);
    }

    #[test]
    fn test_weights_hash_is_deterministic() {
        assert_eq!(weights_hash(&[1.0, 2.0]), weights_hash(&[1.0, 2.0]));
        assert_ne!(weights_hash(&[1.0, 2.0]), weights_hash(&[2.0, 1.0]));
        assert_ne!(weights_hash(&[0.0]), weights_hash(&[-0.0]));
    }
}
