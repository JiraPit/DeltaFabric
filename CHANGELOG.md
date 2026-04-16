# Changelog

This document tracks changes, deviations, and improvements across versions.

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

