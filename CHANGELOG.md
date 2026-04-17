# Changelog

This document tracks changes, deviations, and improvements across versions.

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

| Package | Version |
|---------|---------|
| rkyv | 0.8.15 |
| zenoh | 1.9.0 |
| burn | 0.20.1 |
| burn-ndarray | 0.20.1 (dev) |
| tokio | 1.52.0 |
| anyhow | 1.0.102 |
| tracing | 0.1.44 |
