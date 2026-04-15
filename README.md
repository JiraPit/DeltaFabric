# DeltaFabric

A decentralized, gossip-based synchronization protocol for distributed ML training.

## Core Concepts

### Anchor-Active Weight Model

We maintain two weight buffers per node:
- **W_active**: weights currently being optimized by local gradients
- **W_anchor**: weights representing the last acknowledged state

The delta between these buffers captures the local updates discovered since the last synchronization round. This two-buffer design eliminates the need to store copies of every peer's model, reducing memory overhead compared to traditional decentralized approaches.

### Compressed Delta

To reduce network overhead, we transmit only a sparse subset of the weight changes. Rather than magnitude-based thresholds (which vary across training phases), we select the top K% of changed indices by absolute magnitude. This provides a fixed information budget per synchronization step.

### Built-in Networking

We include a built-in networking layer supporting arbitrary network topologies with zero-copy serialization. Nodes discover peers and exchange deltas without external coordination.

### Keyed Delta Propagation

Each delta is tagged with an originator ID and sequence number. This enables any node to relay updates without creating feedback loops: a node only processes a delta if its sequence number exceeds the last seen value for that originator.

### Dual Update Mechanism

When applying received deltas, both weight buffers are updated. The anchor update ensures the node does not re-broadcast received updates as its own contributions in subsequent rounds.