#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge multiple NS-GBS dataset npz files into one."""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge NS-GBS datasets")
    parser.add_argument("inputs", nargs="+", help="Input npz files")
    parser.add_argument("--out", required=True, help="Output npz file")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()

    all_features, all_labels, all_actions, all_deltas, all_t_idx, all_step = [], [], [], [], [], []

    for path in args.inputs:
        data = np.load(path, allow_pickle=True)
        all_features.extend(data["features"])
        all_labels.extend(data["labels"])
        all_actions.extend(data["actions"])
        all_deltas.extend(data["deltas"])
        all_t_idx.extend(data["t_idx"])
        all_step.extend(data["step"])
        print(f"[Merge] Loaded {len(data['features'])} samples from {path}")

    features = np.array(all_features, dtype=object)
    labels = np.array(all_labels, dtype=np.int64)
    actions = np.array(all_actions, dtype=object)
    deltas = np.array(all_deltas, dtype=object)
    t_idx = np.array(all_t_idx, dtype=np.int32)
    step = np.array(all_step, dtype=np.int32)

    if args.shuffle:
        np.random.seed(args.seed)
        idx = np.random.permutation(len(features))
        features, labels, actions, deltas, t_idx, step = (
            features[idx], labels[idx], actions[idx], deltas[idx], t_idx[idx], step[idx]
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, labels=labels, actions=actions,
                        deltas=deltas, t_idx=t_idx, step=step)
    print(f"[Merge] Saved {len(features)} samples to {out_path}")


if __name__ == "__main__":
    main()
