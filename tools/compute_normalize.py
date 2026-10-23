"""Minimal reference demo for computing the normalization values used by JsonDataset.

Example:
    python tools/compute_normalize.py data/json1.json data/json2.json -o normalize.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mibot.utils.io import compose_action, compose_state, get_value, rotm2aa_batch


ACTION_LENGTH = 30


def json_files(paths):
    files = []
    for path in paths:
        path = Path(path)
        matches = [path] if path.is_file() else glob.glob(str(path / "**/*.json"), recursive=True)
        files.extend(Path(match) for match in matches)
    if not files:
        raise ValueError("no JSON files found")
    return sorted(set(files))


def future(traj, key, frame, steps):
    return np.asarray(get_value(traj, key)[frame : frame + steps], dtype=np.float32)


def frame(traj, key, index):
    return np.asarray(get_value(traj, key)[index], dtype=np.float32)


def pad(values, length):
    if len(values) == length:
        return values
    return np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)])


def make_action(traj, index, length):
    steps = min(length, int(traj["num_frames"]) - index)
    parts = []
    for arm in ("left", "right"):
        rotation = frame(traj, f"proprios.{arm}_ee_rotm", index).reshape(3, 3)
        position = frame(traj, f"proprios.{arm}_ee_pos", index)
        target_position = future(traj, f"actions.{arm}_ee_pos", index, steps)
        target_rotation = future(traj, f"actions.{arm}_ee_rotm", index, steps)
        target_rotation = target_rotation.reshape(-1, 3, 3)
        parts += [
            pad((rotation.T @ (target_position - position).T).T, length),
            pad(rotm2aa_batch(rotation.T @ target_rotation), length),
            pad(
                future(traj, f"actions.{arm}_gripper_pos", index, steps)
                - frame(traj, f"proprios.{arm}_gripper_pos", index),
                length,
            ),
        ]
    parts += [
        pad(
            future(traj, "actions.waist_pos", index, steps)
            - frame(traj, "proprios.waist_pos", index),
            length,
        ),
        pad(future(traj, "actions.base_vel", index, steps), length),
    ]
    return compose_action(*parts, action_length=length)


def make_state(traj, index):
    return compose_state(
        frame(traj, "proprios.left_gripper_pos", index),
        frame(traj, "proprios.left_arm_joint", index),
        frame(traj, "proprios.right_gripper_pos", index),
        frame(traj, "proprios.right_arm_joint", index),
    )


def compute(paths, length, output=None):
    actions, states = [], []
    for path in tqdm(json_files(paths), desc="Processing JSON", unit="file"):
        if output is not None and path.resolve() == Path(output).resolve():
            continue
        with path.open(encoding="utf-8") as file:
            traj = json.load(file)
        num_frames = int(traj["num_frames"])
        for index in range(num_frames):
            states.append(make_state(traj, index))
        for index in range(num_frames - length + 1):
            actions.append(make_action(traj, index, length))

    if not actions:
        raise ValueError(f"JSON files contain no complete {length}-step action window")
    actions = np.asarray(actions, dtype=np.float32)  # (frames, length, 60)
    states = np.asarray(states, dtype=np.float32)   # (frames, 1, 60)
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "q01": np.quantile(states, 0.01, axis=0).tolist(),
        "q99": np.quantile(states, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON files or directories")
    parser.add_argument("-o", "--output", required=True, help="output JSON path")
    parser.add_argument("--action-length", type=int, default=ACTION_LENGTH)
    args = parser.parse_args()
    if args.action_length <= 0:
        parser.error("--action-length must be positive")
    result = compute(args.paths, args.action_length, args.output)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"wrote {args.output}: mean/std={args.action_length}x60, q01/q99=1x60")


if __name__ == "__main__":
    main()
