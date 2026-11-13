import argparse
import glob
import itertools

import torch


# Compares the final weights saved by each node of a distributed run.
# Usage: python ../compare_weights.py "final_weights_dist3_node_*_gamma_0.02.pt"
def main():
    parser = argparse.ArgumentParser(
        description="Pairwise distance between final node weights"
    )
    parser.add_argument("pattern", help="Glob pattern matching the saved .pt files")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 files, found {len(files)}: {files}")

    flats = {}
    for path in files:
        state = torch.load(path, map_location="cpu")
        flats[path] = torch.cat(
            [v.flatten().double() for v in state.values() if v.is_floating_point()]
        )

    mean = torch.stack(list(flats.values())).mean(dim=0)
    print(f"{'node A':45s} {'node B':45s} {'||A-B||':>10s} {'||A-B||/||A||':>14s}")
    relative = []
    for a, b in itertools.combinations(files, 2):
        dist = (flats[a] - flats[b]).norm().item()
        rel = dist / flats[a].norm().item()
        relative.append(rel)
        print(f"{a:45s} {b:45s} {dist:10.4f} {rel:14.6f}")

    spread = sum((f - mean).norm().item() for f in flats.values()) / len(flats)
    print(f"\nMean relative pairwise distance: {sum(relative) / len(relative):.6f}")
    print(f"Mean distance to average model:  {spread:.4f}")


if __name__ == "__main__":
    main()
