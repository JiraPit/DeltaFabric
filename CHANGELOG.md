# Changelog

This document tracks changes, deviations, and improvements across versions.

---

## Unreleased

### Sync Algorithm (breaking)

The previous rule (apply `alpha` × peer deltas to both buffers, relay with `alpha^hops`) had no force pulling models together: on 3-node MNIST the final weights differed by 11.2% versus 14.2% with no syncing at all. Sync now follows CHOCO-SGD while keeping the delta-only design:

- **New `W_mix` buffer**: weighted sum of own and neighbor anchors, rebuilt from received deltas. Received deltas no longer touch `W_active`/`W_anchor`
- **Mixing step**: `W_active += gamma * (W_mix - W_anchor)` at every sync, with Metropolis weights from the discovered graph (preserves the cluster average)
- **Config**: `alpha` and `relay_threshold` replaced by `gamma` (default 0.02). Keep `gamma <= 2 * delta_selection_ratio` (`Config::max_stable_gamma`); larger values diverge. `Config::validate` rejects out-of-range values
- **No relays**: each packet carries only the sender's delta; the origin comes from the topic. Links are undirected and neighbors subscribe to each other
- **Core `SyncState`**: framework-independent protocol state shared by Burn and PyTorch
- Delta generation skips non-finite values instead of panicking on NaN; the sequence number only advances when a delta is sent

### Compression

- **Indices**: sent as LEB128 gaps between sorted positions (about 1 byte each at 1% density, was 4)
- **Values**: new `value_encoding` config: `F32` (default, 4 bytes), `Int8` (1 byte, scaled by the largest value) or `Sign` (1 bit, scaled by the mean absolute value). Zero-rounded entries are dropped
- The sender updates its anchor with the decoded values, so rounding error stays in `W_active - W_anchor` and is sent later
- Wire format: `DeltaPacket.delta` is now an `EncodedDelta` (still rkyv); `core::codec` encodes and decodes it. Malformed deltas are ignored and recovered through the resync path

### Reliability

- Deltas must arrive exactly once: a sequence gap or a mismatching initial-weights hash triggers a resync from neighbor anchor snapshots (Zenoh queryable `node/{id}/anchor`)
- Delta publisher uses `CongestionControl::Block` instead of dropping
- Subscribers are declared before a node announces READY, so early deltas are not lost
- `init_fabric` fails if a neighbor finished discovery without this node (one-sided peer lists)

### Usability

- **Cluster names**: `Config::cluster` (default `"default"`) namespaces all topics as `deltafabric/{cluster}/node/{id}/...`, so separate jobs on one network no longer interfere. Examples read `DF_CLUSTER`
- **Python logging**: warnings go to stderr by default (`RUST_LOG` respected); `delta_fabric.init_logging(level)` enables more detail

### Measurement

- `Session::stats()` (Burn) and `fabric.stats()` (PyTorch) report delta and snapshot traffic
- PyTorch examples save final weights and traffic; `examples/compare_weights.py` compares nodes
- PyTorch: integer state (BatchNorm `num_batches_tracked`) is no longer synced; errors include their cause

---

## v0.7

### Bug Fixes

- **Burn**: Fixed compilation error due to non-existent `map_data` method on `Tensor`
  - Replaced with `Tensor::from_data` to create tensors from `TensorData`
  - Preserved gradient tracking in `apply_params` using `is_require_grad()` and `require_grad()`

---

## v0.6

### Optimization

Both Burn and PyTorch backends now skip unnecessary parameter extraction. Parameters are only extracted when needed: first step (anchor initialization), peer updates received, or sync step (local delta generation). Mathematical behavior is unchanged.

### Bug Fixes

- **Burn**: Fixed bug where `anchor_weights` was not initialized on first step when no peer updates and not sync step
- **PyTorch**: Fixed missing `my_id` argument in `process_deltas` call

---

## v0.5

### Multi-Backend Support

Added PyTorch compatibility layer alongside Burn:

- **New `pytorch` feature flag**: `cargo build --features pytorch`
- **New `src/pytorch/mod.rs`**: Pure PyO3 implementation (no tch/libtorch dependency)
- **Consistent API**: `fabric.step(model)` works identically for both backends
- **Python extension**: Builds as `libdelta_fabric.so` for direct Python import

### New Examples

- **Renamed**: `examples/mnist/` → `examples/mnist_burn/` (Burn ML framework)
- **New**: `examples/mnist_pytorch/` - PyTorch equivalents of all examples

  Both use 10% of MNIST (6,000 samples) for faster training:
  - `mnist_single/`: Single-node baseline
  - `mnist_distributed_2/`: 2-node (3,000 samples each)
  - `mnist_distributed_3/`: 3-node (2,000 samples each)

### Model Architecture

Reduced model size for faster training (~75% reduction):

- Conv channels: 32 → 16
- FC hidden: 64 → 32
- Total params: ~61,738 → ~15,642

---

## v0.4

### API Simplification

Simplified the DeltaFabric API to reduce user boilerplate:

- **New `Config::new(peers)` constructor**: Creates config with sensible defaults (alpha=0.5, delta_selection_ratio=0.01, sync_interval=100, relay_threshold=1e-6)

- **`Fabric::step()` takes ownership**: Now takes model by value, returns updated model with synced params applied

- **`Fabric` manages step count internally**: No need to pass step_count to `step()`

- **Helper functions**: `extract_params()` and `apply_params()` for working with model parameters

- **Simplified usage**:

  ```rust
  use delta_fabric::{Config, Fabric};

  let config = Config::new(peers);
  let mut fabric = Fabric::new(node_id, config).await?;

  let mut model: Model<Autodiff<NdArray<f32>>> = Model::new(&device);
  let optimizer = SgdConfig::new().init();

  for batch in dataset {
      let output = model.forward_classification(batch);
      let grads = GradientsParams::from_grads(output.loss.backward(), &model);
      model = optimizer.step(lr, model, grads);

      // DeltaFabric sync - single call, returns updated model
      model = fabric.step(model).await?;
  }
  ```

### New Files

- `src/burn/traits.rs`: Contains `extract_params()` and `apply_params()` helper functions

---

## v0.4 (continued)

### New Examples

Added partitioned distributed training examples with dataset sharding:

- `examples/mnist/mnist_distributed_2/`: 2-node distributed training
  - Node 1: samples 0-29,999
  - Node 2: samples 30,000-59,999

- `examples/mnist/mnist_distributed_3/`: 3-node distributed training
  - Node 1: samples 0-19,999
  - Node 2: samples 20,000-39,999
  - Node 3: samples 40,000-59,999

### Test Coverage

Added comprehensive tests for burn module helpers:

- `test_extract_params_returns_all_params`
- `test_apply_params_updates_weights`
- `test_extract_params_preserves_order`
- `test_apply_params_with_zeros`
- `test_extract_apply_roundtrip`
- `test_different_model_sizes`

---

## v0.3

### New Examples

Added MNIST distributed training example demonstrating DeltaFabric's weight synchronization protocol:

- `examples/mnist/mnist_single/`: Single-node baseline training (no networking)

### Dependencies Added

- Added `burn-vision` feature support for vision datasets

---

## v0.2

### Breaking Changes

- Renamed `FabricConfig` → `Config`
- Renamed `FabricPacket` → `DeltaPacket`
- Renamed `ArchivedFabricPacket` → `ArchivedDeltaPacket`
- Renamed `init_cluster` → `init_fabric`
- Renamed `top_k_pct` → `delta_selection_ratio`
- Renamed `expected_peers` → `peers`

### New Features

- **Zero-Copy rkyv Deserialization**: Replaced `rkyv::from_bytes` with `rkyv::access` in the hot path
  - Added `access_archived_packet()` helper for zero-copy deserialization
  - `process_deltas()` now accepts `&ArchivedDeltaPacket` instead of `&DeltaPacket`
  - Eliminates heap allocations per incoming packet

### Bug Fixes

- Fixed rkyv type compatibility with archived primitive types (`u64_le`, `u32_le`)

---

## v0.1

Initial implementation following SPEC.md and PLAN.md.

### Deviations from SPEC.md and PLAN.md

This document tracks differences between the implementation and the original specifications.

#### 1. rkyv Zero-Copy Deserialization

**Status:** Not implemented (planned for v2)

**SPEC.md specifies:**

```rust
rkyv::access::<ArchivedFabricPacket, rkyv::rancor::Error>(&payload[..])
```

**Current implementation:**

```rust
rkyv::from_bytes::<FabricPacket, _>(&payload)
```

**Reason:** Using `from_bytes` for simplicity. Zero-copy requires:

- `rkyv::access` to get zero-copy reference
- `process_deltas` accepting `&ArchivedFabricPacket` instead of `&FabricPacket`
- Careful lifetime management

**Planned fix:** Update `process_deltas` signature and use zero-copy access pattern.

---

#### 2. Zenoh Subscriber Pattern

**Status:** Functional equivalent with different implementation

**SPEC.md specifies:**

```rust
delta_subscribers: HashMap<u64, zenoh::Subscriber<'static, ()>>
// ...
while let Ok(sample) = sub.try_recv()
```

**Current implementation:**

```rust
delta_samples: Arc<Mutex<Vec<zenoh::sample::Sample>>>
// Callbacks push to buffer
// pull_packets() drains the buffer
```

**Reason:** Zenoh 1.9's `Subscriber<Handler>` is generic and cannot be stored in a homogeneous HashMap without type erasure. The callback+buffer pattern achieves the same functional result.

---

#### 3. rkyv Deserialize API

**Status:** API difference, functionally correct

**PLAN.md specifies:**

```rust
archived.deserialize(&mut rkyv::rancor::UndisclosedError)
```

**Current implementation:**

```rust
deserialize::<FabricPacket, rkyv::rancor::Error>(archived)
```

**Reason:** rkyv 0.8 uses function-style `deserialize()` instead of method-style.

---

#### 4. Test Coverage

**Status:** Extended from plan

- Added `test_process_deltas_fresh_delta`
- Added `test_process_deltas_stale_delta`
- Added `test_process_deltas_threshold`
- Added `test_end_to_end_delta_flow`
- Added `test_seen_table_deduplication`
- Added `test_flatten_unflatten_preserves_params`

All tests use imported functions with dummy data, no logic reimplementation.

---

### Dependencies Used

| Package      | Version      |
| ------------ | ------------ |
| rkyv         | 0.8.15       |
| zenoh        | 1.9.0        |
| burn         | 0.20.1       |
| burn-ndarray | 0.20.1 (dev) |
| tokio        | 1.52.0       |
| anyhow       | 1.0.102      |
| tracing      | 0.1.44       |
