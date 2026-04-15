#[cfg(test)]
mod networking_tests {
    #[test]
    fn test_node_state_serialization() {
        use super::*;
        use serde_json;

        let state = NodeState {
            expected_peers: vec![1, 2, 3],
            status: "READY".to_string(),
        };

        let json = serde_json::to_string(&state).unwrap();
        let parsed: NodeState = serde_json::from_str(&json).unwrap();

        assert_eq!(state, parsed);
    }
}
