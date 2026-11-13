#[cfg(test)]
mod integration_tests {
    //! In-memory clusters of SyncStates exchanging real serialized packets.

    use crate::core::Config;
    use crate::core::codec::ValueEncoding;
    use crate::core::packet::{DeltaPacket, decode_packet, decode_snapshot, to_bytes};
    use crate::core::sync::{MixingWeights, ReceiveOutcome, SyncState, metropolis_weights};
    use std::collections::{HashMap, HashSet};

    struct Cluster {
        ids: Vec<u64>,
        edges: HashMap<u64, HashSet<u64>>,
        weights: HashMap<u64, MixingWeights>,
        states: HashMap<u64, SyncState>,
        active: HashMap<u64, Vec<f32>>,
    }

    impl Cluster {
        /// Builds a cluster where every node starts from `init`.
        fn new(links: &[(u64, u64)], gamma: f32, ratio: f32, init: &[f32]) -> Self {
            Self::with_encoding(links, gamma, ratio, init, ValueEncoding::F32)
        }

        /// Builds a cluster whose nodes send values with `encoding`.
        fn with_encoding(
            links: &[(u64, u64)],
            gamma: f32,
            ratio: f32,
            init: &[f32],
            encoding: ValueEncoding,
        ) -> Self {
            let mut edges: HashMap<u64, HashSet<u64>> = HashMap::new();
            for &(a, b) in links {
                edges.entry(a).or_default().insert(b);
                edges.entry(b).or_default().insert(a);
            }
            let mut ids: Vec<u64> = edges.keys().copied().collect();
            ids.sort_unstable();

            let weights: HashMap<u64, MixingWeights> = ids
                .iter()
                .map(|&id| (id, metropolis_weights(id, &edges)))
                .collect();
            let mut states = HashMap::new();
            let mut active = HashMap::new();
            for &id in &ids {
                let mut state =
                    SyncState::new(gamma, ratio, weights[&id].clone()).with_encoding(encoding);
                state.initialize(init);
                states.insert(id, state);
                active.insert(id, init.to_vec());
            }
            Self {
                ids,
                edges,
                weights,
                states,
                active,
            }
        }

        /// Runs one sync on every node and delivers every packet to its neighbors.
        /// `drop` lists (origin, receiver) links whose packet is lost this round.
        fn round(&mut self, drop: &[(u64, u64)]) -> Vec<(u64, u64, ReceiveOutcome)> {
            let mut sent: Vec<(u64, Vec<u8>)> = Vec::new();
            for &id in &self.ids {
                let active = self.active.get_mut(&id).unwrap();
                if let Some(packet) = self.states.get_mut(&id).unwrap().sync_step(active).unwrap() {
                    sent.push((id, to_bytes(&packet).unwrap()));
                }
            }

            let mut outcomes = Vec::new();
            for (origin, bytes) in &sent {
                for &to in &self.edges[origin] {
                    if drop.contains(&(*origin, to)) {
                        continue;
                    }
                    let packet: DeltaPacket = decode_packet(bytes).unwrap();
                    let outcome = self.states.get_mut(&to).unwrap().receive(*origin, packet);
                    outcomes.push((*origin, to, outcome));
                }
            }
            outcomes
        }

        /// Serves every missing snapshot through real serialization.
        fn serve_snapshots(&mut self) {
            for &id in &self.ids {
                for peer in self.states[&id].missing_snapshots() {
                    let bytes = to_bytes(&self.states[&peer].snapshot()).unwrap();
                    let snapshot = decode_snapshot(&bytes).unwrap();
                    self.states
                        .get_mut(&id)
                        .unwrap()
                        .receive_snapshot(peer, snapshot)
                        .unwrap();
                }
            }
        }

        /// Applies one gradient step toward each node's own target.
        fn train(&mut self, targets: &HashMap<u64, Vec<f32>>, lr: f32) {
            for (id, active) in self.active.iter_mut() {
                for (w, t) in active.iter_mut().zip(&targets[id]) {
                    *w -= lr * (*w - t);
                }
            }
        }

        /// Max over nodes of |mix_sum - (true weighted sum of anchors)|.
        fn invariant_error(&self) -> f32 {
            let mut worst: f32 = 0.0;
            for &id in &self.ids {
                let w = &self.weights[&id];
                let state = &self.states[&id];
                for (i, &sum) in state.mix_sum().iter().enumerate() {
                    let mut expected = w.self_weight * state.anchor()[i];
                    for (peer, weight) in &w.neighbors {
                        expected += weight * self.states[peer].anchor()[i];
                    }
                    worst = worst.max((sum - expected).abs());
                }
            }
            worst
        }

        fn mean_active(&self) -> Vec<f32> {
            let n = self.ids.len() as f32;
            let len = self.active[&self.ids[0]].len();
            (0..len)
                .map(|i| self.ids.iter().map(|id| self.active[id][i]).sum::<f32>() / n)
                .collect()
        }

        /// Average distance of each node's active weights from the cluster mean.
        fn disagreement(&self) -> f32 {
            let mean = self.mean_active();
            let total: f32 = self
                .ids
                .iter()
                .map(|id| {
                    self.active[id]
                        .iter()
                        .zip(&mean)
                        .map(|(a, m)| (a - m).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .sum();
            total / self.ids.len() as f32
        }
    }

    const MESH: &[(u64, u64)] = &[(1, 2), (1, 3), (2, 3)];
    const RING: &[(u64, u64)] = &[(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)];

    /// Deterministic pseudo-random values in [-1, 1].
    fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn targets(ids: &[u64], len: usize) -> HashMap<u64, Vec<f32>> {
        ids.iter().map(|&id| (id, pseudo_random(id, len))).collect()
    }

    #[test]
    fn test_mix_sum_tracks_true_anchors() {
        for links in [MESH, RING] {
            let mut cluster = Cluster::new(links, 0.5, 0.1, &[0.0; 50]);
            let targets = targets(&cluster.ids, 50);
            for _ in 0..30 {
                for _ in 0..5 {
                    cluster.train(&targets, 0.1);
                }
                cluster.round(&[]);
                assert!(cluster.invariant_error() < 1e-4);
            }
        }
    }

    #[test]
    fn test_nodes_reach_consensus_and_keep_the_average() {
        // One burst of different local training, then sync only.
        let mut cluster = Cluster::new(RING, 0.5, 0.2, &[0.0; 40]);
        let targets = targets(&cluster.ids, 40);
        for _ in 0..20 {
            cluster.train(&targets, 0.2);
        }
        let mean_before = cluster.mean_active();
        let disagreement_before = cluster.disagreement();

        for _ in 0..300 {
            cluster.round(&[]);
        }

        assert!(cluster.disagreement() < 0.01 * disagreement_before);
        for (a, b) in cluster.mean_active().iter().zip(&mean_before) {
            assert!((a - b).abs() < 1e-4, "mixing must preserve the average");
        }
    }

    const LOSSY: [ValueEncoding; 2] = [ValueEncoding::Int8, ValueEncoding::Sign];

    #[test]
    fn test_lossy_encodings_keep_mix_sum_exact() {
        for encoding in LOSSY {
            let mut cluster = Cluster::with_encoding(RING, 0.2, 0.1, &[0.0; 50], encoding);
            let targets = targets(&cluster.ids, 50);
            for _ in 0..30 {
                for _ in 0..5 {
                    cluster.train(&targets, 0.1);
                }
                cluster.round(&[]);
                assert!(cluster.invariant_error() < 1e-4, "{encoding:?}");
            }
        }
    }

    #[test]
    fn test_lossy_encodings_reach_consensus() {
        for encoding in LOSSY {
            let mut cluster = Cluster::with_encoding(RING, 0.4, 0.2, &[0.0; 40], encoding);
            let targets = targets(&cluster.ids, 40);
            for _ in 0..20 {
                cluster.train(&targets, 0.2);
            }
            let mean_before = cluster.mean_active();
            let before = cluster.disagreement();

            for _ in 0..600 {
                cluster.round(&[]);
            }

            assert!(cluster.disagreement() < 0.01 * before, "{encoding:?}");
            for (a, b) in cluster.mean_active().iter().zip(&mean_before) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "{encoding:?} must preserve the average"
                );
            }
        }
    }

    #[test]
    fn test_lossy_encodings_match_f32_while_training() {
        // Same setup as test_mixing_beats_independent_training_with_different_data,
        // at gamma = 2x ratio: compression must not cost consensus.
        let run = |gamma: f32, encoding| {
            let mut cluster = Cluster::with_encoding(MESH, gamma, 0.05, &[0.0; 200], encoding);
            let targets = targets(&cluster.ids, 200);
            for _ in 0..200 {
                for _ in 0..5 {
                    cluster.train(&targets, 0.05);
                }
                cluster.round(&[]);
            }
            cluster.disagreement()
        };

        let independent = run(0.0, ValueEncoding::F32);
        let exact = run(0.1, ValueEncoding::F32);
        assert!(exact < 0.8 * independent);
        for encoding in LOSSY {
            let mixed = run(0.1, encoding);
            assert!(
                (mixed - exact).abs() < 0.1 * exact,
                "{encoding:?}: {mixed} vs f32 {exact}"
            );
        }
    }

    #[test]
    fn test_default_config_converges() {
        // Guards the default gamma: too large for 1% top-k makes nodes diverge.
        let config = Config::default();
        let mut cluster = Cluster::new(
            RING,
            config.gamma,
            config.delta_selection_ratio,
            &[0.0; 300],
        );
        let targets = targets(&cluster.ids, 300);
        for _ in 0..20 {
            cluster.train(&targets, 0.2);
        }
        let before = cluster.disagreement();

        for _ in 0..3000 {
            cluster.round(&[]);
        }

        assert!(cluster.disagreement() < 0.01 * before);
    }

    #[test]
    fn test_too_large_gamma_diverges() {
        // Documents why max_stable_gamma exists.
        let mut cluster = Cluster::new(MESH, 0.5, 0.01, &[0.0; 300]);
        let targets = targets(&cluster.ids, 300);
        for _ in 0..20 {
            cluster.train(&targets, 0.2);
        }
        let before = cluster.disagreement();

        for _ in 0..500 {
            cluster.round(&[]);
        }

        let after = cluster.disagreement();
        assert!(after.is_nan() || after >= before, "{after} < {before}");
    }

    #[test]
    fn test_gamma_zero_never_pulls_nodes_together() {
        let mut cluster = Cluster::new(MESH, 0.0, 0.2, &[0.0; 40]);
        let targets = targets(&cluster.ids, 40);
        for _ in 0..20 {
            cluster.train(&targets, 0.2);
        }
        let before = cluster.disagreement();

        for _ in 0..100 {
            cluster.round(&[]);
        }

        assert!((cluster.disagreement() - before).abs() < 1e-5);
    }

    #[test]
    fn test_mixing_beats_independent_training_with_different_data() {
        // Each node keeps training toward its own target (non-IID data).
        let run = |gamma: f32| {
            let mut cluster = Cluster::new(MESH, gamma, 0.05, &[0.0; 200]);
            let targets = targets(&cluster.ids, 200);
            for _ in 0..200 {
                for _ in 0..5 {
                    cluster.train(&targets, 0.05);
                }
                cluster.round(&[]);
            }
            cluster.disagreement()
        };

        let independent = run(0.0);
        let mixed = run(0.5);
        assert!(
            mixed < 0.5 * independent,
            "mixed {mixed} vs independent {independent}"
        );
    }

    #[test]
    fn test_lost_packet_triggers_resync_that_restores_invariant() {
        let mut cluster = Cluster::new(MESH, 0.5, 0.1, &[0.0; 30]);
        let targets = targets(&cluster.ids, 30);
        cluster.train(&targets, 0.3);
        cluster.round(&[]);

        // Node 1's packet to node 2 is lost.
        cluster.train(&targets, 0.3);
        cluster.round(&[(1, 2)]);
        assert!(!cluster.states[&2].is_resyncing());
        assert!(cluster.invariant_error() > 1e-3);

        // The next packet from node 1 reveals the gap.
        cluster.train(&targets, 0.3);
        let outcomes = cluster.round(&[]);
        assert!(outcomes.contains(&(1, 2, ReceiveOutcome::ResyncStarted)));
        assert!(cluster.states[&2].is_resyncing());

        cluster.serve_snapshots();
        assert!(!cluster.states[&2].is_resyncing());
        assert!(cluster.invariant_error() < 1e-4);

        for _ in 0..10 {
            cluster.train(&targets, 0.3);
            cluster.round(&[]);
            assert!(cluster.invariant_error() < 1e-4);
        }
    }

    #[test]
    fn test_different_initial_weights_trigger_resync() {
        let mut cluster = Cluster::new(MESH, 0.2, 0.1, &[0.0; 20]);
        // Node 3 started from different weights than it was assumed to.
        let other_init = pseudo_random(99, 20);
        cluster.states.get_mut(&3).unwrap().initialize(&other_init);
        cluster.active.insert(3, other_init);
        // It is only detected once node 3 has trained and sends its first delta.
        let targets = targets(&cluster.ids, 20);
        cluster.train(&targets, 0.3);

        let outcomes = cluster.round(&[]);
        assert!(outcomes.contains(&(3, 1, ReceiveOutcome::ResyncStarted)));
        assert!(outcomes.contains(&(3, 2, ReceiveOutcome::ResyncStarted)));

        cluster.serve_snapshots();
        assert!(cluster.invariant_error() < 1e-4);
        for _ in 0..200 {
            cluster.round(&[]);
        }
        assert!(cluster.disagreement() < 1e-2);
    }
}
