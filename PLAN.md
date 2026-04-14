# DeltaFabric: Implementation Plan

This document outlines the complete implementation plan for DeltaFabric, a decentralized, gossip-based synchronization layer for high-performance machine learning.

---

## Project Overview

**Project Name:** DeltaFabric  
**Type:** Rust Library / Framework Integration  
**Core Functionality:** Synchronous Relay Fabric with loop prevention and Batch Aggregation for distributed ML training

---

## Project Structure

```
delta_fabric/
├── Cargo.toml
└── src/
    ├── lib.rs                          # Main library entry point
    │
    ├── core/                           # Core framework (framework-agnostic)
    │   ├── mod.rs                      # Module declarations
    │   │
    │   ├── packet/                     # Communication packet types
    │   │   ├── mod.rs                 # FabricPacket, SparseDelta exports
    │   │   └── test.rs                 # Serialization round-trip tests
    │   │
    │   ├── sync/                       # Synchronization operations
    │   │   ├── mod.rs                 # Module declarations (apply, generate)
    │   │   ├── apply.rs               # process_deltas, apply_deltas (Dual Update Rule)
    │   │   ├── generate.rs            # generate_local_delta
    │   │   └── test.rs                # Unit tests for delta operations
    │   │
    │   ├── networking/                 # Zenoh transport abstraction
    │   │   ├── mod.rs                 # Session, Node, NodeState
    │   │   └── test.rs                 # Mock network tests
    │   │
    │   ├── config/                     # Configuration
    │   │   ├── mod.rs                 # FabricConfig
    │   │   └── test.rs                 # Config builder tests
    │   │
    │   └── test.rs                     # Integration tests (multi-module)
    │
    └── burn/                          # Burn framework integration
        ├── mod.rs                     # Fabric struct, flatten/unflatten
        ├── flatten.rs                 # Parameter extraction
        ├── unflatten.rs               # Parameter reconstruction
        └── test.rs                    # Burn integration tests
```

---

## Dependencies

### Cargo.toml

```toml
[package]
name = "delta_fabric"
version = "0.1.0"
edition = "2021"

[dependencies]
rkyv = { version = "<newest>", features = ["derive"] }
zenoh = "<newest>"
burn = "<newest>"
tokio = { version = "<newest>", features = ["full"] }
serde = { version = "<newest>", features = ["derive"] }
serde_json = "<newest>"
anyhow = "<newest>"
tracing = "<newest>"
tracing-subscriber = { version = "<newest>", features = ["env-filter"] }

[dev-dependencies]
```

---

## Module Specifications

### 1. core/packet

**Purpose:** rkyv-archived types for network transmission.

**Files:**
- `mod.rs`
- `test.rs`

**Types:**
- `FabricPacket` - HashMap<u64, SparseDelta>
- `SparseDelta` - sequence_id, indices, values
- `ArchivedFabricPacket` - type alias for rkyv deserialization

---

### 2. core/sync

**Purpose:** Core synchronization logic.

**Files:**
- `mod.rs` - Module declarations
- `apply.rs` - Delta processing and application (Dual Update Rule)
- `generate.rs` - Local delta generation
- `test.rs`

**Functions:**
- `process_deltas()` - Process incoming packets, aggregate updates, prepare relay
- `apply_deltas()` - Apply aggregated deltas to active and anchor weights
- `generate_local_delta()` - Top-K delta generation from active/anchor weights

---

### 3. core/networking

**Purpose:** Zenoh transport abstraction.

**Files:**
- `mod.rs`
- `test.rs`

**Types:**
- `Node` - id, expected_peers
- `NodeState` - expected_peers, status
- `Session` - Zenoh session wrapper with cluster management

**Methods:**
- `Session::new()` - Open Zenoh session
- `Session::init_cluster()` - Barrier synchronization
- `Session::pull_packets()` - Receive pending packets
- `Session::broadcast()` - Send packet (rkyv serialized)
- `Session::shutdown()` - Graceful departure

---

### 4. core/config

**Purpose:** Configuration management.

**Files:**
- `mod.rs`
- `test.rs`

**Types:**
- `FabricConfig` - alpha, top_k_pct, sync_interval, relay_threshold, expected_peers

---

### 5. burn

**Purpose:** Burn framework integration.

**Files:**
- `mod.rs`
- `flatten.rs`
- `unflatten.rs`
- `test.rs`

**Types:**
- `Fabric<B>` - Main fabric struct with session and config

**Functions:**
- `flatten_burn_model()` - Extract model parameters to Vec<f32>
- `unflatten_burn_model()` - Write Vec<f32> back to model

**Methods:**
- `Fabric::new()` - Initialize fabric with node_id and config
- `Fabric::step()` - Synchronization step
- `Fabric::shutdown()` - Cleanup

---

## Implementation Steps

### Step 1: Project Setup

```bash
cargo new delta_fabric --lib
cd delta_fabric
cargo add rkyv --features derive
cargo add zenoh
cargo add burn
cargo add tokio --features full
cargo add serde --features derive
cargo add serde_json
```

### Step 2: core/packet

**`src/core/packet/mod.rs`**

```rust
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct FabricPacket {
    pub updates: HashMap<u64, SparseDelta>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct SparseDelta {
    pub sequence_id: u64,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

pub type ArchivedFabricPacket = rkyv::Archived<FabricPacket>;
```

**`src/core/packet/test.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena, Deserialize};

    #[test]
    fn test_fabric_packet_roundtrip() {
        let packet = FabricPacket {
            updates: HashMap::from([
                (1, SparseDelta {
                    sequence_id: 42,
                    indices: vec![0, 1, 2],
                    values: vec![0.1, 0.2, 0.3],
                }),
            ]),
        };

        let mut arena = Arena::new();
        let bytes = to_bytes_with_alloc::<_, rkyv::rancor::Error>(&packet, arena.acquire()).unwrap();
        let archived = rkyv::access::<ArchivedFabricPacket, _>(&bytes[..]).unwrap();
        let deserialized: FabricPacket = archived.deserialize(&mut rkyv::rancor::UndisclosedError).unwrap();

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
```

### Step 3: core/sync

**`src/core/sync/mod.rs`**

```rust
pub mod apply;
pub mod generate;

pub use apply::{process_deltas, apply_deltas};
pub use generate::generate_local_delta;
```

**`src/core/sync/apply.rs`**

```rust
use crate::core::packet::{ArchivedFabricPacket, SparseDelta};
use tracing::{debug, trace};
use std::collections::HashMap;

pub fn process_deltas(
    aggregator: &mut HashMap<u32, f32>,
    incoming: &ArchivedFabricPacket,
    seen_table: &mut HashMap<u64, u64>,
    alpha: f32,
    relay_threshold: f32,
) -> Option<HashMap<u64, SparseDelta>> {
    let mut relay_updates = HashMap::new();
    let mut fresh_count = 0;
    let mut stale_count = 0;

    for (&origin_id, delta) in incoming.updates.iter() {
        let last_seq = seen_table.get(&origin_id).unwrap_or(&0);

        if delta.sequence_id > *last_seq {
            fresh_count += 1;
            seen_table.insert(origin_id, delta.sequence_id);
            trace!(origin_id = %origin_id, seq = %delta.sequence_id, "Processing fresh delta");

            let mut relay_indices = Vec::new();
            let mut relay_values = Vec::new();

            for (i, &idx) in delta.indices.iter().enumerate() {
                let val = delta.values[i];
                let damped_val = val * alpha;

                *aggregator.entry(idx).or_insert(0.0) += damped_val;

                if damped_val.abs() >= relay_threshold {
                    relay_indices.push(idx);
                    relay_values.push(damped_val);
                }
            }

            if !relay_indices.is_empty() {
                relay_updates.insert(origin_id, SparseDelta {
                    sequence_id: delta.sequence_id,
                    indices: relay_indices,
                    values: relay_values,
                });
            }
        } else {
            stale_count += 1;
            trace!(origin_id = %origin_id, seq = %delta.sequence_id, last_seq = %last_seq, "Skipping stale delta");
        }
    }

    if fresh_count > 0 || stale_count > 0 {
        debug!(
            fresh = %fresh_count,
            stale = %stale_count,
            relay_count = %relay_updates.len(),
            "Processed relay packet"
        );
    }

    if relay_updates.is_empty() { None } else { Some(relay_updates) }
}

/// Applies aggregated deltas to both active weights and anchor weights.
///
/// This is the "Dual Update Rule" - updating anchor prevents re-broadcasting
/// of acknowledged peer updates.
///
/// # Arguments
/// * `active` - The active weight buffer (modified in place)
/// * `anchor` - The anchor/reference weight buffer (modified in place)
/// * `aggregator` - HashMap of index -> accumulated delta values
pub fn apply_deltas(
    active: &mut [f32],
    anchor: &mut [f32],
    aggregator: &HashMap<u32, f32>,
) {
    for (&idx, &delta) in aggregator.iter() {
        active[idx as usize] += delta;
        anchor[idx as usize] += delta;
        trace!(idx = %idx, delta = %delta, "Applied delta to weights");
    }
}
```

**`src/core/sync/generate.rs`**

```rust
use crate::core::packet::SparseDelta;
use tracing::{debug, trace};

/// Generates local delta by selecting Top-K largest changes from active weights.
///
/// # Arguments
/// * `active` - Current active weights
/// * `anchor` - Anchor/reference weights (updated to acknowledge sent delta)
/// * `top_k_pct` - Percentage of weights to share (0.01 = 1%)
/// * `my_id` - This node's ID (for packet keying)
/// * `seq` - Sequence number for this delta
///
/// # Returns
/// SparseDelta containing top-K indices and values, or None if no significant changes
pub fn generate_local_delta(
    active: &[f32],
    anchor: &mut [f32],
    top_k_pct: f32,
    _my_id: u64,
    seq: u64,
) -> Option<SparseDelta> {
    let mut deltas: Vec<(u32, f32)> = active.iter()
        .zip(anchor.iter())
        .enumerate()
        .map(|(i, (act, anc))| (i as u32, act - anc))
        .collect();

    deltas.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    let k = (active.len() as f32 * top_k_pct).ceil() as usize;
    let top_k = &deltas[..k.min(deltas.len())];

    let mut indices = Vec::new();
    let mut values = Vec::new();

    for &(idx, val) in top_k {
        if val != 0.0 {
            indices.push(idx);
            values.push(val);
            anchor[idx as usize] += val;
        }
    }

    if indices.is_empty() { None } else {
        Some(SparseDelta { sequence_id: seq, indices, values })
    }
}
```

**`src/core/sync/test.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

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
        
        // Top 40% = 2 elements (ceil(5 * 0.4) = 2)
        // Largest values: 5.0 (idx 4), 4.0 (idx 3)
        assert_eq!(result.indices, vec![4, 3]);
        assert_eq!(result.values, vec![5.0, 4.0]);
        
        // Anchor should be updated
        assert_eq!(anchor[4], 5.0);
        assert_eq!(anchor[3], 4.0);
    }

    #[test]
    fn test_process_deltas_fresh_delta() {
        // Create archived packet with sequence_id > seen_table
        // Test that fresh deltas are processed
        // Test that stale deltas are ignored
    }

    #[test]
    fn test_process_deltas_threshold() {
        // Test that values below threshold are not relayed
    }
}
```

### Step 4: core/networking

**`src/core/networking/mod.rs`**

```rust
use std::collections::{HashMap, HashSet, VecDeque};
use anyhow::{Context, Result};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error};
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
    delta_subscribers: HashMap<u64, zenoh::Subscriber<'static, ()>>,
    pub delta_publisher: Option<zenoh::publication::Publisher<'static>>,
}

impl Session {
    pub async fn new(node: Node) -> Result<Self> {
        let zenoh = zenoh::open(zenoh::Config::default())
            .await
            .context("Failed to open Zenoh session")?;

        info!(node_id = %node.id, "Opened Zenoh session");

        Ok(Self {
            zenoh,
            node,
            delta_subscribers: HashMap::new(),
            delta_publisher: None,
        })
    }

    pub async fn init_cluster(&mut self) -> Result<()> {
        let my_topic = format!("node/{}/state", self.node.id);
        let publisher = self.zenoh.declare_publisher(&my_topic)
            .await
            .context("Failed to declare state publisher")?;

        let state_sub = self.zenoh.declare_subscriber("node/*/state")
            .await
            .context("Failed to declare state subscriber")?;

        info!(node_id = %self.node.id, peers = ?self.node.expected_peers, "Starting cluster initialization");

        let mut known_states: HashMap<u64, NodeState> = HashMap::new();
        let mut global_nodes: HashSet<u64> = HashSet::new();
        let mut is_ready = false;

        loop {
            let my_state = NodeState {
                expected_peers: self.node.expected_peers.clone(),
                status: if is_ready { "READY" } else { "DISCOVERING" }.to_string(),
            };

            let state_json = serde_json::to_string(&my_state)
                .context("Failed to serialize NodeState")?;

            publisher.put(state_json)
                .await
                .context("Failed to publish state")?;

            while let Ok(sample) = state_sub.try_recv() {
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
                if state.status == "OFFLINE" { continue; }
                for &peer in &state.expected_peers {
                    edges.entry(id).or_default().insert(peer);
                    edges.entry(peer).or_default().insert(id);
                }
            }

            let (mut visited, mut queue, mut missing) = (HashSet::new(), VecDeque::new(), false);
            queue.push_back(self.node.id);

            while let Some(curr) = queue.pop_front() {
                if !visited.insert(curr) { continue; }
                match known_states.get(&curr) {
                    Some(s) if s.status != "OFFLINE" => {
                        if let Some(neighbors) = edges.get(&curr) {
                            for &n in neighbors { queue.push_back(n); }
                        }
                    }
                    _ => { missing = true; break; }
                }
            }

            if !missing && !is_ready {
                global_nodes = visited.clone();
                is_ready = true;
                info!(node_count = %global_nodes.len(), "Discovery complete, waiting for READY signals");
            }

            if is_ready && global_nodes.iter().all(|id| {
                known_states.get(id).map_or(false, |s| s.status == "READY")
            }) {
                break;
            }

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        for peer_id in &self.node.expected_peers {
            let sub = self.zenoh.declare_subscriber(format!("node/{}/delta", peer_id))
                .await
                .context("Failed to declare delta subscriber")?;
            self.delta_subscribers.insert(*peer_id, sub);
        }

        self.delta_publisher = Some(
            self.zenoh.declare_publisher(format!("node/{}/delta", self.node.id))
                .await
                .context("Failed to declare delta publisher")?
        );

        info!(node_id = %self.node.id, "Cluster initialization complete");

        Ok(())
    }

    pub fn pull_packets(&self) -> Vec<zenoh::sample::Sample> {
        let mut samples = Vec::new();
        for sub in self.delta_subscribers.values() {
            while let Ok(sample) = sub.try_recv() {
                samples.push(sample);
            }
        }
        samples
    }

    pub async fn broadcast(&self, packet: crate::core::packet::FabricPacket) -> Result<()> {
        use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena};
        
        let mut arena = Arena::new();
        let bytes = to_bytes_with_alloc::<_, rkyv::rancor::Error>(
            &packet,
            arena.acquire()
        ).context("Serialization of FabricPacket failed")?;

        if let Some(publ) = &self.delta_publisher {
            publ.put(bytes.to_vec())
                .await
                .context("Zenoh broadcast failed")?;
        }

        Ok(())
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let state = NodeState {
            expected_peers: self.node.expected_peers.clone(),
            status: "OFFLINE".to_string(),
        };

        let state_json = serde_json::to_string(&state)
            .context("Failed to serialize OFFLINE state")?;

        match self.zenoh.put(
            format!("node/{}/state", self.node.id),
            state_json
        ).await {
            Ok(_) => {
                info!(node_id = %self.node.id, "Broadcast OFFLINE status");
            }
            Err(e) => {
                error!(node_id = %self.node.id, error = %e, "Failed to broadcast OFFLINE status");
            }
        }

        self.delta_subscribers.clear();
        self.delta_publisher = None;

        Ok(())
    }
}
```

**`src/core/networking/test.rs`**

```rust
#[cfg(test)]
mod tests {
    // Mock tests for session logic
    // Integration tests require actual Zenoh setup
    
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
```

### Step 5: core/config

**`src/core/config/mod.rs`**

```rust
#[derive(Debug, Clone)]
pub struct FabricConfig {
    pub alpha: f32,
    pub top_k_pct: f32,
    pub sync_interval: u64,
    pub relay_threshold: f32,
    pub expected_peers: Vec<u64>,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            top_k_pct: 0.01,
            sync_interval: 100,
            relay_threshold: 1e-6,
            expected_peers: vec![],
        }
    }
}

impl FabricConfig {
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn top_k_pct(mut self, top_k_pct: f32) -> Self {
        self.top_k_pct = top_k_pct;
        self
    }

    pub fn sync_interval(mut self, sync_interval: u64) -> Self {
        self.sync_interval = sync_interval;
        self
    }

    pub fn relay_threshold(mut self, relay_threshold: f32) -> Self {
        self.relay_threshold = relay_threshold;
        self
    }

    pub fn expected_peers(mut self, expected_peers: Vec<u64>) -> Self {
        self.expected_peers = expected_peers;
        self
    }
}
```

**`src/core/config/test.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FabricConfig::default();
        
        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.top_k_pct, 0.01);
        assert_eq!(config.sync_interval, 100);
        assert_eq!(config.relay_threshold, 1e-6);
        assert!(config.expected_peers.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let config = FabricConfig::default()
            .alpha(0.3)
            .top_k_pct(0.02)
            .sync_interval(50)
            .relay_threshold(1e-5)
            .expected_peers(vec![1, 2, 3]);
        
        assert_eq!(config.alpha, 0.3);
        assert_eq!(config.top_k_pct, 0.02);
        assert_eq!(config.sync_interval, 50);
        assert_eq!(config.relay_threshold, 1e-5);
        assert_eq!(config.expected_peers, vec![1, 2, 3]);
    }
}
```

### Step 6: core/mod.rs

**`src/core/mod.rs`**

```rust
pub mod config;
pub mod networking;
pub mod packet;
pub mod sync;

pub use config::FabricConfig;
pub use networking::{Node, NodeState, Session};
pub use packet::{ArchivedFabricPacket, FabricPacket, SparseDelta};
pub use sync::{apply_deltas, generate_local_delta, process_deltas};
```

### Step 7: core/test.rs (Integration Tests)

**`src/core/test.rs`**

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_end_to_end_delta_flow() {
        // 1. Generate local delta
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut anchor = vec![0.0; 5];
        
        let delta = generate_local_delta(&active, &mut anchor, 0.4, 1, 1).unwrap();
        assert_eq!(delta.indices.len(), 2);
        
        // 2. Create packet
        let mut packet = FabricPacket { updates: HashMap::new() };
        packet.updates.insert(1, delta);
        
        // 3. Process packet (simulate receiving)
        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut seen_table: HashMap<u64, u64> = HashMap::new();
        
        // Note: ArchivedFabricPacket requires rkyv serialization first
        // This is a conceptual test showing the flow
    }
    
    #[test]
    fn test_seen_table_deduplication() {
        // Test that stale deltas are rejected by seen_table
    }
}
```

### Step 8: burn/flatten.rs

**`src/burn/flatten.rs`**

```rust
use anyhow::{Context, Result};
use tracing::debug;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

struct ParamCollector {
    data: Vec<f32>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let values: Vec<f32> = param.val()
            .to_data()
            .value
            .into_iter()
            .collect();
        self.data.extend(values);
    }
}

pub fn flatten_burn_model<B: Backend, M: Module<B>>(model: &M) -> Result<Vec<f32>> {
    let num_params = model.num_params();
    debug!(num_params = %num_params, "Flattening model parameters");

    let mut collector = ParamCollector { data: Vec::new() };
    model.visit(&mut collector);

    debug!(num_elements = %collector.data.len(), "Flattened model to vector");
    Ok(collector.data)
}
```

### Step 9: burn/unflatten.rs

**`src/burn/unflatten.rs`**

```rust
use anyhow::{Context, Result};
use tracing::{debug, warn};
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

struct ParamSetter {
    data: Vec<f32>,
    pos: usize,
}

impl<B: Backend> ModuleMapper<B> for ParamSetter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let num_elements = param.val().num_elements();
        let end_pos = self.pos + num_elements;

        if end_pos > self.data.len() {
            warn!(
                expected = %num_elements,
                available = %(self.data.len() - self.pos),
                "Insufficient data for parameter"
            );
        }

        let tensor_data = self.data[self.pos..end_pos].to_vec();
        self.pos = end_pos;

        let new_tensor = Tensor::from_floats(
            burn::tensor::TensorData::new(tensor_data, param.val().dims()),
            &param.val().device(),
        );

        param.map(|_| new_tensor)
    }
}

pub fn unflatten_burn_model<B: Backend, M: Module<B>>(
    model: &mut M,
    flat_data: &[f32]
) -> Result<()> {
    let num_params = model.num_params();
    debug!(num_params = %num_params, "Unflattening model parameters");

    let mut setter = ParamSetter {
        data: flat_data.to_vec(),
        pos: 0,
    };
    
    model.map(&mut setter);

    debug!(
        pos = %setter.pos,
        expected = %flat_data.len(),
        "Unflatten complete"
    );

    if setter.pos != flat_data.len() {
        warn!(
            consumed = %setter.pos,
            expected = %flat_data.len(),
            "Not all data was consumed during unflatten"
        );
    }

    Ok(())
}
```

### Step 10: burn/mod.rs

**`src/burn/mod.rs`**

```rust
pub mod flatten;
pub mod unflatten;

use crate::core::{
    config::FabricConfig,
    networking::Session,
    packet::FabricPacket,
    sync::{apply_deltas, generate_local_delta, process_deltas},
};
use crate::core::packet::ArchivedFabricPacket;
use anyhow::{Context, Result};
use tracing::{info, warn, error};
use burn::module::Module;
use burn::tensor::backend::Backend;
use std::collections::HashMap;

pub use flatten::flatten_burn_model;
pub use unflatten::unflatten_burn_model;

pub struct Fabric<B: Backend> {
    pub session: Session,
    pub config: FabricConfig,
    pub anchor_weights: Vec<f32>,
    pub seen_table: HashMap<u64, u64>,
    pub local_sequence: u64,
}

impl<B: Backend> Fabric<B> {
    pub async fn new(node_id: u64, config: FabricConfig) -> Result<Self> {
        info!(node_id = %node_id, "Initializing DeltaFabric");

        let node = crate::core::networking::Node {
            id: node_id,
            expected_peers: config.expected_peers.clone(),
        };

        let mut session = Session::new(node).await
            .context("Failed to create session")?;
        
        session.init_cluster().await
            .context("Failed to initialize cluster")?;

        info!(node_id = %node_id, "DeltaFabric initialized successfully");

        Ok(Self {
            session,
            config,
            anchor_weights: Vec::new(),
            seen_table: HashMap::new(),
            local_sequence: 0,
        })
    }

    pub async fn step<M: Module<B>>(&mut self, mut model: M, step_count: u64) -> Result<M> {
        let mut active_flat = flatten_burn_model(&model)
            .context("Failed to flatten model")?;

        if self.anchor_weights.is_empty() {
            self.anchor_weights = active_flat.clone();
            info!(
                step = %step_count,
                num_weights = %active_flat.len(),
                "Initialized anchor weights"
            );
        }

        let mut aggregator: HashMap<u32, f32> = HashMap::new();
        let mut relay_packet = FabricPacket { updates: HashMap::new() };

        // 1. COLLECT & AGGREGATE PEER DELTAS
        for sample in self.session.pull_packets() {
            let payload = sample.payload().to_bytes();
            if let Ok(incoming) = rkyv::access::<ArchivedFabricPacket, rkyv::rancor::Error>(&payload[..]) {
                if let Some(updates) = process_deltas(
                    &mut aggregator,
                    &incoming,
                    &mut self.seen_table,
                    self.config.alpha,
                    self.config.relay_threshold,
                ) {
                    relay_packet.updates.extend(updates);
                }
            } else {
                warn!("Failed to deserialize incoming packet");
            }
        }

        let num_peer_updates = aggregator.len();
        if num_peer_updates > 0 {
            info!(
                step = %step_count,
                num_updates = %num_peer_updates,
                "Applying peer deltas"
            );
        }

        // 2. BATCH APPLY UPDATES (Dual Update Rule)
        apply_deltas(&mut active_flat, &mut self.anchor_weights, &aggregator);

        // 3. LOCAL TOP-K GENERATION
        if step_count % self.config.sync_interval == 0 {
            self.local_sequence += 1;
            if let Some(delta) = generate_local_delta(
                &active_flat,
                &mut self.anchor_weights,
                self.config.top_k_pct,
                self.session.node.id,
                self.local_sequence,
            ) {
                info!(
                    step = %step_count,
                    seq = %self.local_sequence,
                    num_indices = %delta.indices.len(),
                    "Generated local delta"
                );
                relay_packet.updates.insert(self.session.node.id, delta);
            }
        }

        // 4. BROADCAST
        if !relay_packet.updates.is_empty() {
            self.session.broadcast(relay_packet).await
                .context("Failed to broadcast packet")?;
        }

        unflatten_burn_model(&mut model, &active_flat)
            .context("Failed to unflatten model")?;

        Ok(model)
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        info!(node_id = %self.session.node.id, "Shutting down DeltaFabric");
        
        self.session.shutdown().await
            .context("Failed to shutdown session")?;

        info!(node_id = %self.session.node.id, "DeltaFabric shutdown complete");
        Ok(())
    }
}
```

**`src/burn/test.rs`**

```rust
#[cfg(test)]
mod tests {
    // Note: Full tests require burn with ndarray backend for no_std compatibility
    
    #[test]
    fn test_flatten_unflatten_preserves_params() {
        // Integration test with actual model
    }
}
```

### Step 11: lib.rs

**`src/lib.rs`**

```rust
pub mod core;
pub mod burn;

use anyhow::Result;

pub use core::{FabricConfig, Session, Node, NodeState, FabricPacket, SparseDelta, ArchivedFabricPacket, process_deltas};
pub use burn::Fabric;

pub mod prelude {
    pub use anyhow::{Context, Result};
}

/// Initialize the tracing subscriber for logging.
///
/// Call this once at the start of your application before using DeltaFabric.
/// 
/// # Example
/// ```rust
/// use delta_fabric::init_tracing;
/// 
/// fn main() {
///     init_tracing();
///     // ... rest of application
/// }
/// ```
pub fn init_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,delta_fabric=debug"));
    
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

/// Initialize tracing with a custom filter.
///
/// # Arguments
/// * `filter` - A string in the format expected by EnvFilter (e.g., "info,delta_fabric=debug")
pub fn init_tracing_with_filter(filter: &str) -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};
    
    let filter = EnvFilter::try_from(filter)
        .context("Invalid tracing filter")?;
    
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    Ok(())
}
```

---

## Implementation Order

| Step | Module | Files | Description |
|------|--------|-------|-------------|
| 1 | Setup | `Cargo.toml` | Project init, dependencies |
| 2 | core/packet | 2 files | Packet types with rkyv |
| 3 | core/sync | 4 files | apply.rs (process_deltas, apply_deltas), generate.rs + mod.rs + test.rs |
| 4 | core/networking | 2 files | Zenoh session with error handling |
| 5 | core/config | 2 files | Configuration |
| 6 | core | 1 file | Module declarations |
| 7 | core | 1 file | Integration tests |
| 8 | burn | 4 files | Burn integration with error handling |
| 9 | root | 1 file | lib.rs with tracing init |

---

## Logging and Error Handling Conventions

### Logging Levels

| Level | Usage |
|-------|-------|
| `error!` | Operation failed (returned Err) |
| `warn!` | Operation succeeded but with unexpected conditions |
| `info!` | Major lifecycle events (init, shutdown, cluster join) |
| `debug!` | Significant state changes (delta processed, applied) |
| `trace!` | Low-level details (individual weight updates) |

### Error Handling

All public async functions return `Result<T>` using `anyhow`:

```rust
pub async fn public_function() -> Result<ReturnType> {
    some_operation()
        .await
        .context("Description of what failed")?;
    
    another_operation()
        .await
        .map_err(|e| anyhow!("Failed: {}", e))?;
    
    Ok(return_value)
}
```

---

## Testing Strategy

| Level | Location | Scope |
|-------|----------|-------|
| Unit | `*/test.rs` | Individual functions |
| Module | `*/test.rs` | Single module behavior |
| Integration | `core/test.rs` | Multi-module interaction |
| Integration | `burn/test.rs` | Framework integration |

---

## Appendix: Official Examples

These examples demonstrate patterns from each library's official documentation.

### A. rkyv (Zero-Copy Serialization)

*rkyv allows zero-copy deserialization by memory-mapping bytes directly without unpacking.*

```rust
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
struct Test {
    int: u8,
    string: String,
    option: Option<Vec<i32>>,
}

fn main() {
    let value = Test {
        int: 42,
        string: "hello world".to_string(),
        option: Some(vec![1, 2, 3, 4]),
    };

    use rkyv::{api::high::to_bytes_with_alloc, ser::allocator::Arena};
    let mut arena = Arena::new();
    let bytes = to_bytes_with_alloc::<_, rkyv::rancor::Error>(&value, arena.acquire()).unwrap();

    let archived = rkyv::access::<rkyv::Archived<Test>, rkyv::rancor::Error>(&bytes[..]).unwrap();
    assert_eq!(archived, &value);
}
```

### B. Zenoh (P2P Mesh Networking)

*Zenoh provides a lightweight P2P pub/sub transport suitable for low-latency distributed training.*

#### Publisher Example

```rust
use std::time::Duration;
use zenoh::{bytes::Encoding, Config};

#[tokio::main]
async fn main() {
    zenoh::init_log_from_env_or("error");

    let session = zenoh::open(Config::default()).await.unwrap();
    let key_expr = "node/1/delta";
    let publisher = session.declare_publisher(key_expr).await.unwrap();

    for idx in 0..u32::MAX {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let buf = format!("[{idx:4}] delta data");

        publisher
            .put(buf)
            .encoding(Encoding::TEXT_PLAIN)
            .await
            .unwrap();
    }
}
```

#### Subscriber Example

```rust
use zenoh::Config;

#[tokio::main]
async fn main() {
    zenoh::init_log_from_env_or("error");

    let session = zenoh::open(Config::default()).await.unwrap();
    let key_expr = "node/*/delta";
    let subscriber = session.declare_subscriber(key_expr).await.unwrap();

    while let Ok(sample) = subscriber.recv_async().await {
        let payload = sample.payload().to_bytes();
        println!("Received delta from '{}'", sample.key_expr().as_str());
    }
}
```

### C. Burn (Deep Learning Framework)

*Burn provides a flexible deep learning framework with hardware acceleration support.*

```rust
use burn::{
    nn::{
        BatchNorm, PaddingConfig2d,
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([1, 64], [3, 3], device, true);
        let conv2 = ConvBlock::new([64, 64], [3, 3], device, true);
        let hidden_size = 64 * 5 * 5;
        let fc1 = nn::LinearConfig::new(hidden_size, 128).init(device);
        let fc2 = nn::LinearConfig::new(128, 128).init(device);
        let fc3 = nn::LinearConfig::new(128, 10).init(device);
        let dropout = nn::DropoutConfig::new(0.25).init();

        Self {
            conv1,
            conv2,
            dropout,
            fc1,
            fc2,
            fc3,
            activation: nn::Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B>,
    pool: Option<MaxPool2d>,
    activation: nn::Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device, pool: bool) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);
        let pool = if pool {
            Some(MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init())
        } else {
            None
        };

        Self { conv, norm, pool, activation: nn::Relu::new() }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = self.activation.forward(x);

        if let Some(pool) = &self.pool {
            pool.forward(x)
        } else {
            x
        }
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = burn::train::MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
```

---

## Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Damping factor for peer updates |
| `top_k_pct` | 0.01 (1%) | Percentage of weights to share |
| `sync_interval` | 100 | Steps between local delta generation |
| `relay_threshold` | 1e-6 | Minimum delta magnitude for relay |

---

## Key Design Decisions

1. **Two-Buffer System (Anchor-Active)** - Minimizes memory overhead while preventing feedback loops
2. **Keyed Deltas** - Originator_ID prevents loops, Sequence_ID ensures freshness
3. **Batch Aggregation** - Accumulates peer updates before applying to model weights
4. **Dual Anchor Update** - Updates both active and anchor to prevent re-broadcasting acknowledged data
5. **Top-K Selection** - Adaptive sparsity regardless of gradient magnitude
6. **Framework-Agnostic Core** - Core logic separated from Burn for potential PyTorch/TF support
