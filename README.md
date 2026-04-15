# DeltaFabric

A decentralized weight synchronization protocol for distributed ML training with built-in networking. Currently supporting PyTorch and Burn.

## Core Concepts

### No-Master Architecture

Eliminates master-server bottlenecks by letting every node participate equally in synchronization. It also makes life easier by cutting out the need to set up or manage a central server.

### Built-in Networking and Automatic Peer Discovery

We include a built-in networking layer powered by the `Zenoh` P2P protocol. Nodes discover peers and exchange deltas automatically without external coordination.

### Anchor-Active Weight Model

We maintain two weight buffers per node:

- **W_active**: weights currently being optimized by local gradients
- **W_anchor**: weights representing the last acknowledged state

The delta between these buffers captures the local updates discovered since the last synchronization round. This two-buffer design eliminates the need to store copies of every peer's model, reducing memory overhead compared to traditional decentralized approaches.

### Compressed Delta

Sending an entire set of weights becomes impractically large as the model grows. To reduce bandwidth and increase speed, we only send the most significant changes (deltas) per step. This allows a single packet to hold not only one delta but also others relayed from additional peers.
