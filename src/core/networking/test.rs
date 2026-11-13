#[cfg(test)]
mod networking_tests {
    //! Live Zenoh tests. Node IDs are unique per test (and far from the IDs
    //! used by the examples) because all sessions on a host discover each other.

    use super::super::*;
    use crate::core::codec::{ValueEncoding, encode_delta};
    use crate::core::exchange;
    use crate::core::packet::{DeltaPacket, SparseDelta, decode_packet, decode_snapshot};
    use crate::core::sync::SyncState;

    fn test_packet(sequence_id: u64) -> DeltaPacket {
        let delta = SparseDelta {
            sequence_id,
            indices: vec![0, 5, 9],
            values: vec![0.5, -1.0, 2.0],
        };
        DeltaPacket {
            delta: encode_delta(&delta, ValueEncoding::F32),
            init_hash: None,
        }
    }

    async fn ready_session(id: u64, peers: Vec<u64>) -> Session {
        let mut session = Session::new(Node {
            id,
            peers,
            cluster: "test".to_string(),
        })
        .await
        .unwrap();
        session.init_fabric().await.unwrap();
        session
    }

    async fn ready_pair(
        a: u64,
        a_peers: Vec<u64>,
        b: u64,
        b_peers: Vec<u64>,
    ) -> (Session, Session) {
        tokio::join!(ready_session(a, a_peers), ready_session(b, b_peers))
    }

    /// Polls pull_packets until at least `count` packets arrive or the timeout expires.
    async fn wait_for_packets(session: &Session, count: usize) -> Vec<(u64, Vec<u8>)> {
        let mut received = Vec::new();
        for _ in 0..200 {
            received.extend(session.pull_packets());
            if received.len() >= count {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        received
    }

    #[test]
    fn test_node_state_serialization() {
        let state = NodeState {
            peers: vec![1, 2, 3],
            status: "READY".to_string(),
            neighbors: vec![1, 2, 3],
        };

        let json = serde_json::to_string(&state).unwrap();
        let parsed: NodeState = serde_json::from_str(&json).unwrap();

        assert_eq!(state, parsed);
    }

    async fn ready_in_cluster(cluster: &str, id: u64, peers: Vec<u64>) -> Session {
        let mut session = Session::new(Node {
            id,
            peers,
            cluster: cluster.to_string(),
        })
        .await
        .unwrap();
        session.init_fabric().await.unwrap();
        session
    }

    #[test]
    fn test_cluster_key() {
        assert_eq!(
            cluster_key("job-a", 7, "delta"),
            "deltafabric/job-a/node/7/delta"
        );
        assert_eq!(
            cluster_key("job-a", "*", "state"),
            "deltafabric/job-a/node/*/state"
        );
        // A cluster literally named "node" must not confuse ID parsing.
        assert_eq!(node_id_from_key(&cluster_key("node", 9, "delta")), Some(9));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_clusters_with_same_ids_are_isolated() {
        // Two jobs both use node IDs 1201 and 1202 on the same host.
        let (a1, a2, b1, b2) = tokio::join!(
            ready_in_cluster("iso-a", 1201, vec![1202]),
            ready_in_cluster("iso-a", 1202, vec![1201]),
            ready_in_cluster("iso-b", 1201, vec![1202]),
            ready_in_cluster("iso-b", 1202, vec![1201]),
        );

        a1.broadcast(test_packet(1)).await.unwrap();
        assert_eq!(wait_for_packets(&a2, 1).await.len(), 1);

        // Give a stray packet time to arrive in the other cluster, then check none did.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(b1.pull_packets().is_empty());
        assert!(b2.pull_packets().is_empty());
        assert_eq!(b2.stats().packets_received, 0);
    }

    #[test]
    fn test_node_id_from_key() {
        assert_eq!(node_id_from_key("deltafabric/test/node/42/delta"), Some(42));
        assert_eq!(node_id_from_key("node/abc/delta"), None);
        assert_eq!(node_id_from_key("node"), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_stats_start_at_zero() {
        let session = Session::new(Node {
            id: 1101,
            peers: vec![],
            cluster: "test".to_string(),
        })
        .await
        .unwrap();

        assert_eq!(session.stats(), TrafficStats::default());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_isolated_node_has_no_neighbors() {
        let session = ready_session(1102, vec![]).await;

        assert!(session.neighbors().is_empty());
        assert_eq!(session.mixing_weights(), MixingWeights::isolated());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_broadcast_counts_sent_bytes() {
        let session = ready_session(1103, vec![]).await;

        let packet = test_packet(1);
        let expected_len = to_bytes(&packet).unwrap().len() as u64;

        session.broadcast(packet).await.unwrap();
        session.broadcast(test_packet(2)).await.unwrap();

        let stats = session.stats();
        assert_eq!(stats.packets_sent, 2);
        assert_eq!(stats.bytes_sent, 2 * expected_len);
        assert_eq!(stats.packets_received, 0);
        assert_eq!(stats.bytes_received, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_two_nodes_deliver_packets_with_origin() {
        let (a, b) = ready_pair(1104, vec![1105], 1105, vec![1104]).await;

        let packet = test_packet(1);
        let expected_len = to_bytes(&packet).unwrap().len() as u64;
        a.broadcast(packet.clone()).await.unwrap();

        let received = wait_for_packets(&b, 1).await;
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].0, 1104);
        assert_eq!(decode_packet(&received[0].1).unwrap(), packet);

        let stats_b = b.stats();
        assert_eq!(stats_b.packets_received, 1);
        assert_eq!(stats_b.bytes_received, expected_len);
        assert_eq!(a.stats().bytes_sent, expected_len);
    }

    #[test]
    fn test_node_state_without_neighbors_field_parses() {
        let parsed: NodeState = serde_json::from_str(r#"{"peers":[2],"status":"READY"}"#).unwrap();
        assert!(parsed.neighbors.is_empty());
    }

    fn ready_state(neighbors: Vec<u64>) -> NodeState {
        NodeState {
            peers: vec![],
            status: "READY".to_string(),
            neighbors,
        }
    }

    #[test]
    fn test_neighbor_agreement_accepts_symmetric_links() {
        let states = HashMap::from([(2, ready_state(vec![1, 3])), (3, ready_state(vec![2, 1]))]);
        assert!(check_neighbor_agreement(1, &[2, 3], &states).is_ok());
        assert!(check_neighbor_agreement(1, &[], &states).is_ok());
    }

    #[test]
    fn test_neighbor_agreement_rejects_one_sided_link() {
        // Node 2 finished discovery alone and does not know about node 1.
        let states = HashMap::from([(2, ready_state(vec![])), (3, ready_state(vec![1]))]);
        let err = check_neighbor_agreement(1, &[2, 3], &states).unwrap_err();
        assert!(err.to_string().contains("Node 2"));

        assert!(check_neighbor_agreement(1, &[4], &states).is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_one_sided_peer_list_starting_together() {
        // Only 1106 lists 1107. Depending on timing, 1107 either discovers 1106
        // (link works both ways) or finishes alone (1106 must fail, not mix one-sidedly).
        let (a, b) = tokio::join!(
            async {
                let mut s = Session::new(Node {
                    id: 1106,
                    peers: vec![1107],
                    cluster: "test".to_string(),
                })
                .await
                .unwrap();
                let result = s.init_fabric().await;
                (s, result)
            },
            ready_session(1107, vec![])
        );
        let (a, a_result) = a;

        if b.neighbors() == vec![1106] {
            a_result.unwrap();
            assert_eq!(a.neighbors(), vec![1107]);
            assert_eq!(a.mixing_weights().neighbors[&1107], 0.5);
            assert_eq!(b.mixing_weights().neighbors[&1106], 0.5);
        } else {
            assert!(b.neighbors().is_empty());
            assert!(a_result.is_err());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_snapshot_request_and_reply() {
        let (mut a, b) = ready_pair(1108, vec![1109], 1109, vec![1108]).await;
        let snapshot = AnchorSnapshot {
            sequence_id: 3,
            anchor: vec![1.0, 2.0, 3.0],
        };

        a.request_snapshot(1109).await.unwrap();
        // A second request while one is pending is not sent.
        a.request_snapshot(1109).await.unwrap();

        let mut replied = 0;
        let mut received = Vec::new();
        for _ in 0..200 {
            for query in b.pull_snapshot_requests() {
                b.reply_snapshot(query, &snapshot).await.unwrap();
                replied += 1;
            }
            received.extend(a.poll_snapshots());
            if !received.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        assert_eq!(replied, 1);
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].0, 1109);
        assert_eq!(decode_snapshot(&received[0].1).unwrap(), snapshot);
        assert_eq!(a.stats().snapshots_received, 1);
        assert_eq!(b.stats().snapshots_sent, 1);
        assert_eq!(
            a.stats().snapshot_bytes_received,
            b.stats().snapshot_bytes_sent
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_exchange_recovers_lost_packet_over_network() {
        let (mut a, mut b) = ready_pair(1110, vec![1111], 1111, vec![1110]).await;
        let mut state_a = SyncState::new(0.5, 1.0, a.mixing_weights());
        let mut state_b = SyncState::new(0.5, 1.0, b.mixing_weights());
        state_a.initialize(&[0.0, 0.0]);
        state_b.initialize(&[0.0, 0.0]);

        // A sends seq 1, which B loses.
        let first = state_a.sync_step(&mut [1.0, 0.0]).unwrap().unwrap();
        a.broadcast(first).await.unwrap();
        assert_eq!(wait_for_packets(&b, 1).await.len(), 1);

        // A sends seq 2; B detects the gap and resyncs through snapshots.
        let second = state_a.sync_step(&mut [1.0, 4.0]).unwrap().unwrap();
        a.broadcast(second).await.unwrap();

        for _ in 0..200 {
            exchange(&mut b, &mut state_b).await.unwrap();
            exchange(&mut a, &mut state_a).await.unwrap();
            if !state_b.is_resyncing() && b.stats().snapshots_received == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        assert_eq!(b.stats().snapshots_received, 1);
        assert!(!state_b.is_resyncing());
        // B's mix_sum = 0.5 * own anchor (0, 0) + 0.5 * A's anchor
        let expected: Vec<f32> = state_a.anchor().iter().map(|a| 0.5 * a).collect();
        assert_eq!(state_b.mix_sum(), expected.as_slice());
        assert_eq!(state_a.local_sequence(), 2);
    }
}
