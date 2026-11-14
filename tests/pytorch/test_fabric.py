"""Tests for the PyTorch binding.

Run after building the extension (maturin develop --features pytorch):
    python tests/pytorch/test_fabric.py
or with pytest. Node IDs are far from the ones used by the examples, because
all DeltaFabric nodes on a host discover each other.
"""

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn as nn
from delta_fabric import Config, Fabric


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.norm = nn.BatchNorm1d(3)

    def forward(self, x):
        return self.norm(self.fc(x))


def float_vector(model):
    return torch.cat(
        [v.flatten() for v in model.state_dict().values() if v.is_floating_point()]
    )


# Trains one node: same start, a node-specific push, then sync-only steps.
def run_node(node_id, peer_id, push, out_dir, value_encoding="f32"):
    torch.manual_seed(0)
    model = Net()
    config = Config(
        peers=[peer_id],
        gamma=0.5,
        delta_selection_ratio=1.0,
        sync_interval=1,
        value_encoding=value_encoding,
    )
    fabric = Fabric(node_id=node_id, config=config)

    model = fabric.step(model)
    with torch.no_grad():
        for p in model.parameters():
            p += push
    model.norm.num_batches_tracked.fill_(node_id)

    for _ in range(60):
        model = fabric.step(model)
        time.sleep(0.02)

    torch.save(
        {
            "weights": float_vector(model),
            "num_batches_tracked": model.norm.num_batches_tracked.item(),
            "stats": fabric.stats(),
        },
        os.path.join(out_dir, f"node_{node_id}.pt"),
    )
    fabric.close()


def test_config_defaults_and_stable_gamma():
    config = Config(peers=[2])
    assert abs(config.gamma - 0.02) < 1e-9
    assert abs(config.delta_selection_ratio - 0.01) < 1e-9
    assert config.sync_interval == 100
    assert abs(config.max_stable_gamma() - 0.02) < 1e-9
    assert config.value_encoding == "f32"


def test_value_encoding_is_validated():
    for name in ["f32", "int8", "sign"]:
        assert Config(peers=[], value_encoding=name).value_encoding == name
    try:
        Config(peers=[], value_encoding="f16")
    except ValueError as e:
        assert "f16" in str(e)
    else:
        raise AssertionError("unknown encoding should be rejected")

    # An invalid name set after construction is caught when the Fabric starts.
    config = Config(peers=[])
    config.value_encoding = "bogus"
    try:
        Fabric(node_id=3104, config=config)
    except RuntimeError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("unknown encoding should be rejected by Fabric")


def test_invalid_config_is_rejected():
    try:
        Fabric(node_id=3100, config=Config(peers=[], gamma=2.0))
    except RuntimeError as e:
        assert "gamma" in str(e)
    else:
        raise AssertionError("gamma=2.0 should be rejected")


def test_isolated_node_reports_stats():
    torch.manual_seed(0)
    model = Net()
    fabric = Fabric(node_id=3101, config=Config(peers=[], sync_interval=1))
    model = fabric.step(model)
    with torch.no_grad():
        model.fc.weight += 1.0
    before = float_vector(model).clone()
    model = fabric.step(model)

    stats = fabric.stats()
    fabric.close()
    assert torch.equal(float_vector(model), before), "no neighbors: weights unchanged"
    assert stats["packets_sent"] == 1
    assert stats["bytes_sent"] > 0
    assert set(stats) == {
        "packets_sent",
        "bytes_sent",
        "packets_received",
        "bytes_received",
        "snapshots_sent",
        "snapshot_bytes_sent",
        "snapshots_received",
        "snapshot_bytes_received",
    }


# Runs Python code in a fresh process (logging can only be set up once per process).
def run_snippet(code):
    env = {k: v for k, v in os.environ.items() if k != "RUST_LOG"}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env
    )


def test_cluster_config():
    assert Config(peers=[]).cluster == "default"
    assert Config(peers=[], cluster="job-a").cluster == "job-a"
    try:
        Fabric(node_id=3105, config=Config(peers=[], cluster="a/b"))
    except RuntimeError as e:
        assert "cluster" in str(e)
    else:
        raise AssertionError("invalid cluster name should be rejected")


def test_warnings_are_visible_by_default():
    result = run_snippet(
        "from delta_fabric import Fabric, Config\n"
        "f = Fabric(node_id=3120, config=Config(peers=[], gamma=0.5))\n"
        "f.close()\n"
    )
    assert result.returncode == 0, result.stderr
    assert "gamma is above the stable range" in result.stderr
    assert "DeltaFabric initialized" not in result.stderr, "info is hidden by default"


def test_init_logging_enables_info_once():
    result = run_snippet(
        "import delta_fabric\n"
        "from delta_fabric import Fabric, Config\n"
        "delta_fabric.init_logging('info')\n"
        "f = Fabric(node_id=3121, config=Config(peers=[]))\n"
        "f.close()\n"
        "try:\n"
        "    delta_fabric.init_logging('debug')\n"
        "except RuntimeError:\n"
        "    print('second call raised')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "DeltaFabric initialized" in result.stderr
    assert "second call raised" in result.stdout


def test_two_nodes_converge():
    for i, value_encoding in enumerate(["f32", "int8", "sign"]):
        two_nodes_converge(3110 + 2 * i, 3111 + 2 * i, value_encoding)


def two_nodes_converge(id_a, id_b, value_encoding):
    torch.manual_seed(0)
    initial = float_vector(Net())

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as out_dir:
        procs = [
            ctx.Process(target=run_node, args=(id_a, id_b, 1.0, out_dir, value_encoding)),
            ctx.Process(target=run_node, args=(id_b, id_a, -1.0, out_dir, value_encoding)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, f"node process failed with {p.exitcode}"

        a = torch.load(os.path.join(out_dir, f"node_{id_a}.pt"))
        b = torch.load(os.path.join(out_dir, f"node_{id_b}.pt"))

    params = sum(p.numel() for p in Net().parameters())
    start_distance = 2.0 * params**0.5
    distance = (a["weights"] - b["weights"]).norm().item()
    assert distance < 0.01 * start_distance, f"{value_encoding}: distance {distance}"

    # Nodes step independently, so a round where one node mixes before the
    # other's delta arrives shifts the average slightly (exact only in lockstep).
    average = (a["weights"] + b["weights"]) / 2
    assert torch.allclose(average, initial, atol=0.25), "mixing keeps the average"

    assert a["num_batches_tracked"] == id_a, "integer buffers are not synced"
    assert b["num_batches_tracked"] == id_b
    assert a["stats"]["packets_received"] > 0
    assert a["stats"]["snapshots_received"] == 0, "no resync expected"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ... ok")
