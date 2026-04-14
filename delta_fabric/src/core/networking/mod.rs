use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::vec::Vec;
use tracing::{error, info, warn};
use zenoh::Session as ZenohSession;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeState {
    pub expected_peers: Vec<u64>,
    pub status: String,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u64,
    pub expected_peers: Vec<u64>,
}

pub struct Session {
    zenoh: ZenohSession,
    pub node: Node,
    delta_samples: Arc<Mutex<Vec<zenoh::sample::Sample>>>,
    delta_publisher: Option<zenoh::pubsub::Publisher<'static>>,
}

impl Session {
    pub async fn new(node: Node) -> Result<Self> {
        let zenoh = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open Zenoh session: {}", e))?;

        info!(node_id = %node.id, "Opened Zenoh session");

        Ok(Self {
            zenoh,
            node,
            delta_samples: Arc::new(Mutex::new(Vec::new())),
            delta_publisher: None,
        })
    }

    pub async fn init_cluster(&mut self) -> Result<()> {
        let my_topic = format!("node/{}/state", self.node.id);
        let publisher = self
            .zenoh
            .declare_publisher(&my_topic)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to declare state publisher: {}", e))?;

        let state_sub = self
            .zenoh
            .declare_subscriber("node/*/state")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to declare state subscriber: {}", e))?;

        info!(node_id = %self.node.id, peers = ?self.node.expected_peers, "Starting cluster initialization");

        let mut known_states: HashMap<u64, NodeState> = HashMap::new();
        let mut global_nodes: HashSet<u64> = HashSet::new();
        let mut is_ready = false;

        loop {
            let my_state = NodeState {
                expected_peers: self.node.expected_peers.clone(),
                status: if is_ready { "READY" } else { "DISCOVERING" }.to_string(),
            };

            let state_json =
                serde_json::to_string(&my_state).context("Failed to serialize NodeState")?;

            publisher
                .put(state_json)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to publish state: {}", e))?;

            let mut samples_to_process = Vec::new();
            while let Ok(sample) = state_sub.recv_async().await {
                samples_to_process.push(sample);
                if samples_to_process.len() > 100 {
                    break;
                }
            }

            for sample in samples_to_process {
                if let Some(id_str) = sample.key_expr().as_str().split('/').nth(1) {
                    if let Ok(incoming_id) = id_str.parse::<u64>() {
                        match serde_json::from_slice::<NodeState>(&sample.payload().to_bytes()) {
                            Ok(state) => {
                                known_states.insert(incoming_id, state);
                            }
                            Err(e) => {
                                warn!(node_id = %incoming_id, error = %e, "Failed to parse NodeState from peer");
                            }
                        }
                    }
                }
            }
            known_states.insert(self.node.id, my_state.clone());

            let mut edges: HashMap<u64, HashSet<u64>> = HashMap::new();
            for (&id, state) in &known_states {
                if state.status == "OFFLINE" {
                    continue;
                }
                for &peer in &state.expected_peers {
                    edges.entry(id).or_default().insert(peer);
                    edges.entry(peer).or_default().insert(id);
                }
            }

            let (mut visited, mut queue, mut missing) = (HashSet::new(), VecDeque::new(), false);
            queue.push_back(self.node.id);

            while let Some(curr) = queue.pop_front() {
                if !visited.insert(curr) {
                    continue;
                }
                match known_states.get(&curr) {
                    Some(s) if s.status != "OFFLINE" => {
                        if let Some(neighbors) = edges.get(&curr) {
                            for &n in neighbors {
                                queue.push_back(n);
                            }
                        }
                    }
                    _ => {
                        missing = true;
                        break;
                    }
                }
            }

            if !missing && !is_ready {
                global_nodes = visited.clone();
                is_ready = true;
                info!(node_count = %global_nodes.len(), "Discovery complete, waiting for READY signals");
            }

            if is_ready
                && global_nodes
                    .iter()
                    .all(|id| known_states.get(id).map_or(false, |s| s.status == "READY"))
            {
                break;
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        self.delta_publisher = Some(
            self.zenoh
                .declare_publisher(format!("node/{}/delta", self.node.id))
                .await
                .map_err(|e| anyhow::anyhow!("Failed to declare delta publisher: {}", e))?,
        );

        let samples_ref = self.delta_samples.clone();
        for peer_id in &self.node.expected_peers {
            let samples = samples_ref.clone();
            self.zenoh
                .declare_subscriber(format!("node/{}/delta", peer_id))
                .callback(move |sample| {
                    if let Ok(mut guard) = samples.lock() {
                        guard.push(sample);
                    }
                })
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to declare delta subscriber for peer {}: {}",
                        peer_id,
                        e
                    )
                })?;
        }

        info!(node_id = %self.node.id, "Cluster initialization complete");

        Ok(())
    }

    pub fn pull_packets(&self) -> Vec<zenoh::sample::Sample> {
        let mut samples = Vec::new();
        if let Ok(mut guard) = self.delta_samples.lock() {
            samples.extend(guard.drain(..));
        }
        samples
    }

    pub async fn broadcast(&self, packet: crate::core::packet::FabricPacket) -> Result<()> {
        use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena};

        let mut arena = Arena::new();
        let bytes = to_bytes_with_alloc::<_, rkyv::rancor::Error>(&packet, arena.acquire())
            .context("Serialization of FabricPacket failed")?;

        if let Some(publ) = &self.delta_publisher {
            publ.put(bytes.to_vec())
                .await
                .map_err(|e| anyhow::anyhow!("Zenoh broadcast failed: {}", e))?;
        }

        Ok(())
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let state = NodeState {
            expected_peers: self.node.expected_peers.clone(),
            status: "OFFLINE".to_string(),
        };

        let state_json =
            serde_json::to_string(&state).context("Failed to serialize OFFLINE state")?;

        match self
            .zenoh
            .put(format!("node/{}/state", self.node.id), state_json)
            .await
        {
            Ok(_) => {
                info!(node_id = %self.node.id, "Broadcast OFFLINE status");
            }
            Err(e) => {
                error!(node_id = %self.node.id, error = %e, "Failed to broadcast OFFLINE status");
            }
        }

        self.delta_publisher = None;

        Ok(())
    }
}
