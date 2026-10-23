# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import copy
import glob
import json
import os
import random
import re
from functools import lru_cache

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
from transformers.utils import logging
from tqdm import tqdm

from mibot.utils.io import (
    build_action_mask,
    compose_action,
    compose_state,
    get_value,
    normalize_action_parts,
    normalize_quantile,
    resize_image,
    rotm2aa_batch,
    validate_quantiles,
    validate_stats,
)

logger = logging.get_logger(__name__)
PROMPT_RE = re.compile(r"(<image>|<video>)")


# _augment resizes every image (history frames included) to this pixel budget, so it caps the
# history resolution at sqrt(AUGMENT_MAX_PIXELS) = 400. Raising it also changes the current-frame
# size; history and current frames are meant to share one size.
AUGMENT_MAX_PIXELS = 160000

# Keys the data config must state explicitly (checked at the start of JsonDataset.__init__).
RECIPE_KEYS = ("batch_size", "history_frames", "history_stride", "history_res", "thumbs_stride",
               "history_pool", "history_dropout", "history_detach")


class JsonDataset(Dataset):
    def __init__(self, params):
        data = params["train_datasets"]
        # Recipe keys must be written out (0 is fine) rather than defaulted: a config without them
        # would silently train a single-frame model with batch 16.
        _missing = [k for k in RECIPE_KEYS if k not in data]
        if _missing:
            raise ValueError(
                f"data config is missing recipe keys {_missing}; write them out explicitly"
                " (e.g. history_frames: 0 for a single-frame model). The Q2 recipe is configs/data/q2.yaml"
            )
        self.swap_red_blue = bool(data.get("swap_red_blue", False))
        self._video_root = data.get("video_root")
        self._thumbs_root = data.get("thumbs_root")
        self.action_length = int(data.get("action_length", params.get("action_length", 30)))
        self.batch_size = int(data.get("batch_size", 16))
        # max_steps counts optimizer steps; each consumes accumulate_grad_batches micro-batches per rank.
        self.max_samples = (int(params.get("max_steps", 1000)) * int(params.get("accumulate_grad_batches", 1))
                            * self.batch_size * int(os.environ.get("WORLD_SIZE", 1)))
        self.mean, self.std = validate_stats(data["mean"], data["std"], self.action_length)
        self.q01, self.q99 = validate_quantiles(data["q01"], data["q99"])
        # --- History frames. All off when history_frames == 0. ---
        self.history_frames = int(data.get("history_frames", 0))          # N_max
        self.history_stride = int(data.get("history_stride", 25))         # anchors at t - stride*k (25 Hz -> ~1 FPS)
        self.history_res = int(data.get("history_res", 128))              # thumbnail max_pixels = res*res
        self.thumbs_stride = int(data.get("thumbs_stride", 5))            # tools/precompute_thumbs.py --stride
        # History dropout: 1 = N ~ U{0..N_max}; 0 = every sample gets min(N_max, t/stride) frames.
        self.history_dropout = int(data.get("history_dropout", 1))
        # Zero preserves the original image grid; 16 enables 4x4 history pooling.
        self.history_pool = int(data.get("history_pool", 0))
        if self.history_pool not in (0, 16):
            raise ValueError(f"history_pool must be 0 (off) or 16, got {self.history_pool}")
        # Whether the history-frame vision path is detached: 0 = gradients flow, 1 = no-grad forward.
        self.history_detach = int(data.get("history_detach", 0))
        # A higher history resolution without pooling grows the sequence past collator.max_length,
        # where the collator would drop samples.
        if self.history_res > 128 and not self.history_pool:
            raise ValueError(
                f"history_res={self.history_res} raises the history resolution without history_pool;"
                " sequences would exceed collator.max_length. Enable both together."
            )
        if self.history_pool and self.history_dropout != 0:
            raise ValueError(
                f"history_pool={self.history_pool} requires history_dropout=0 (fixed history length),"
                f" got history_dropout={self.history_dropout}"
            )
        self.ptp_steps = [int(v) for v in data.get("ptp_steps", [0, 5, 10, 15, 20])]  # per-anchor sub-sampled targets
        self.ptp_length = self.history_frames * len(self.ptp_steps)      # rows of the past segment (5*N_max)
        if self.history_frames and max(self.ptp_steps) >= self.history_stride:
            raise ValueError("ptp_steps must stay below history_stride so adjacent anchors do not overlap")
        self.files = self._json_files(data["paths"])
        self.samples = self._samples()
        if self.history_pool:
            self._validate_history_grid()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        traj = self._read_json(sample["file"])
        frame = sample["frame_index"]
        steps = min(self.action_length, int(traj["num_frames"]) - frame)
        if steps <= 0:
            raise IndexError(f"invalid sample at index={index}: frame_index={frame}")

        prompt = self._prompt(traj)
        # History: N anchors t - stride*k (k = N..1, oldest first).
        # history_dropout=1: N = min(U{0..N_max}, floor(t/stride))
        # history_dropout=0: N = min(N_max, floor(t/stride))
        n_hist = 0
        if self.history_frames:
            budget = random.randint(0, self.history_frames) if self.history_dropout else self.history_frames
            n_hist = min(budget, frame // self.history_stride)
        anchors = [frame - self.history_stride * k for k in range(n_hist, 0, -1)]
        hist_images = self._history_images(sample["file"], anchors)
        if self.history_frames:
            # Instruction first (so history tokens can attend to it under the causal mask), then the
            # history placeholders, then the ORIGINAL user turn unchanged (current views + instruction).
            instr = self._instruction_text(prompt)
            head = f"Task: {instr}\n# History (oldest first, {n_hist} frames)\n" + "".join("<image>" for _ in anchors)
            prompt["conversations"][0]["value"] = head + ("\n" if n_hist else "") + prompt["conversations"][0]["value"]
        left = self._arm_action(traj, "left", frame, steps)
        right = self._arm_action(traj, "right", frame, steps)
        waist = self._delta(traj, "proprios.waist_pos", "actions.waist_pos", frame, steps)
        base = self._pad(self._future(traj, "actions.base_vel", frame, steps), steps)
        action_parts = normalize_action_parts((*left, *right, waist, base), self.mean, self.std)
        action = compose_action(*action_parts, action_length=self.action_length)
        state = normalize_quantile(self._state(traj, frame), self.q01, self.q99)
        action_mask = build_action_mask(self.action_length, self._mask(traj, steps))
        prompt["conversations"] += [
            {"from": "human", "value": "Robot state: <state>"},
            {"from": "gpt", "value": "".join(f"<a_{index}>" for index in range(steps)) + "<score>"},
        ]

        current = self._images(traj, prompt["images"], frame)
        item = {
            "messages": self._messages(prompt["conversations"], self._augment(hist_images + current)),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "state": torch.from_numpy(state),
            "vlm_action_target": torch.from_numpy(action[:steps]),
            "vlm_action_mask": torch.from_numpy(action_mask[:steps]),
            "vlm_action_actual_length": steps,
            # Tells the collator how many leading images are history frames (shortened placeholders,
            # compact-grid positions); 0 when history_pool=0. The collator removes both keys before the model.
            "n_hist": n_hist if self.history_pool else 0,
            "history_pool": self.history_pool,
            "history_detach": self.history_detach,
        }
        if self.history_frames:
            past_action, past_mask = self._past_targets(traj, anchors)
            item["past_action"] = torch.from_numpy(past_action)
            item["past_action_mask"] = torch.from_numpy(past_mask)
        return item

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_json(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def _json_files(paths):
        files = set()
        for path in paths:
            if os.path.isfile(path) and path.endswith(".json"):
                files.add(path)
            elif os.path.isdir(path):
                files.update(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
            else:
                raise ValueError(f"training path is not a JSON file or directory: {path}")
        if not files:
            raise ValueError("training paths contain no JSON files")
        return sorted(files)

    def _samples(self):
        source_samples = []
        logger.info(f"train set has {len(self.files)} files.")
        for path in tqdm(self.files):
            count = int(self._read_json(path)["num_frames"])
            if count <= 0:
                logger.warning(f"File {path} has no frames, skipping.")
                continue
            source_samples.extend({"file": path, "frame_index": i} for i in range(count))

        if not source_samples:
            raise ValueError("training data has no frames")
        q, r = divmod(self.max_samples, len(source_samples))
        samples = source_samples * q + source_samples[:r]
        logger.info(f"Raw train samples: {len(source_samples)}")
        logger.info(f"Total train samples: {len(samples)}")
        return samples

    def _prompt(self, traj):
        prompts = traj.get("instruction", {}).get("general") or []
        if not prompts:
            raise ValueError(f"trajectory {traj.get('time', '<unknown>')} has no instruction.general")
        prompt = copy.deepcopy(random.choice(prompts))
        conversations = prompt.get("conversations", [])
        if len(conversations) >= 2:
            conversations[0]["value"] += " /no_cot"
            conversations[1]["value"] = "<cot></cot>"
        return prompt

    @staticmethod
    def _instruction_text(prompt):
        value = prompt["conversations"][0]["value"]
        marker = "Generate robot actions for the task:"
        tail = value.split(marker, 1)[1] if marker in value else value
        return tail.replace("/no_cot", "").strip()

    # patch_size(16) * spatial_merge_size(2): pixel side covered by one LLM vision token
    _PIXELS_PER_TOKEN = 32

    def _validate_history_grid(self):
        """Reject, before training, a history resolution whose pooling would upsample.

        Reads one real thumbnail and measures its grid; a missing thumbnail cache also fails here.
        """
        side = int(self.history_pool**0.5)
        npy = self._thumbs_path(self.files[0])
        arr = np.load(npy, mmap_mode="r")
        height, width = int(arr.shape[1]), int(arr.shape[2])
        grid = (height // self._PIXELS_PER_TOKEN, width // self._PIXELS_PER_TOKEN)
        survived = resize_image(
            Image.new("RGB", (width, height)), factor=32, max_pixels=AUGMENT_MAX_PIXELS
        ).size
        if survived != (width, height):
            raise ValueError(
                f"history_res={self.history_res} thumbnails {width}x{height} are resized by _augment to "
                f"{survived[0]}x{survived[1]} (AUGMENT_MAX_PIXELS={AUGMENT_MAX_PIXELS}), so the resolution has"
                " no effect. Larger history frames need a larger AUGMENT_MAX_PIXELS, which also changes the"
                " current-frame size."
            )
        if grid[0] < side or grid[1] < side:
            raise ValueError(
                f"history_res={self.history_res} gives {width}x{height} thumbnails, an LLM-side grid "
                f"{grid[0]}x{grid[1]} below the pooling target {side}x{side}, so pooling would upsample."
                " Raise history_res (and recompute the thumbnails) or disable history_pool."
            )

    # Optional node-local copies of videos and thumbnails (video_root / thumbs_root): the path prefix up
    # to a marker is rewritten, the bytes are the same. Unset = the paths in the JSON are used as is.

    @staticmethod
    def _staged(path, root, mark):
        if not root:
            return path
        i = path.find(mark)
        if i < 0:
            raise ValueError(f"staging root {root} is set but {path} has no {mark}")
        return os.path.join(root, path[i + len(mark):])

    def _thumbs_path(self, json_path):
        rel = json_path.replace(os.sep + "data" + os.sep, os.sep + "thumbs" + os.sep, 1)
        return JsonDataset._staged(rel[: -len(".json")] + f".ego{self.history_res}s{self.thumbs_stride}.npy",
                                   self._thumbs_root, "/thumbs/")

    def _history_images(self, json_path, anchors):
        """Head-camera thumbnails for the anchors, from mem_precompute_thumbs.py output
        (<root>/thumbs/<chunk>/<episode>.ego<res>s<stride>.npy, frames 0, stride, 2*stride, ...).
        An anchor a is read at floor(a / thumbs_stride) (<= stride-1 frames early)."""
        if not anchors:
            return []
        arr = self._read_thumbs(self._thumbs_path(json_path))
        return [Image.fromarray(np.asarray(arr[a // self.thumbs_stride])) for a in anchors]

    @staticmethod
    @lru_cache(maxsize=64)
    def _read_thumbs(path):
        return np.load(path, mmap_mode="r")

    def _past_targets(self, traj, anchors):
        """PTP-style sparse multi-anchor reconstruction targets: for each anchor, the
        normalized 30-step action chunk computed at that anchor (relative to the anchor's own pose,
        same pipeline as the current chunk) sub-sampled at ptp_steps. Rows are left-padded to
        ptp_length (oldest anchor first); absent anchors get zero rows and mask 0. Normalization
        reuses mean/std rows [ptp_steps] because stats are stored per within-chunk time step."""
        k = len(self.ptp_steps)
        past = np.zeros((self.ptp_length, self.mean.shape[-1]), dtype=np.float32)
        temporal = np.zeros(self.ptp_length, dtype=np.int32)
        first_slot = self.history_frames - len(anchors)
        for j, a in enumerate(anchors):
            steps = min(self.action_length, int(traj["num_frames"]) - a)
            left = self._arm_action(traj, "left", a, steps)
            right = self._arm_action(traj, "right", a, steps)
            waist = self._delta(traj, "proprios.waist_pos", "actions.waist_pos", a, steps)
            base = self._pad(self._future(traj, "actions.base_vel", a, steps), steps)
            parts = normalize_action_parts((*left, *right, waist, base), self.mean, self.std)
            chunk = compose_action(*parts, action_length=self.action_length)
            valid = self._mask(traj, steps)
            row0 = (first_slot + j) * k
            for i, st in enumerate(self.ptp_steps):
                past[row0 + i] = chunk[st]
                temporal[row0 + i] = int(valid[st])
        return past, build_action_mask(self.ptp_length, temporal)

    def _images(self, traj, keys, frame):
        images = []
        frame = [max(frame, 0)]
        for key in keys:
            infos = get_value(traj, key)
            if infos is None:
                raise ValueError(f"{key} is missing in trajectory {traj.get('time', '<unknown>')}")
            for info in infos:
                video = VideoReader(JsonDataset._staged(info["path"], self._video_root, "/videos/"), num_threads=2)
                video.seek(0)
                images.extend(
                    Image.fromarray(image)
                    for image in JsonDataset._crop_bbox(
                        video.get_batch([frame[0] + int(info.get("start", 0))]).asnumpy(),
                        info.get("crop_bbox"),
                    )
                )
        return images

    @staticmethod
    def _crop_bbox(frames, bbox):
        if bbox is None:
            return frames
        height, width = frames.shape[1:3]
        top, left, crop_height, crop_width = (int(value) for value in bbox)
        output = np.zeros((frames.shape[0], crop_height, crop_width, frames.shape[3]), dtype=frames.dtype)
        source_top, source_left = max(0, top), max(0, left)
        source_bottom, source_right = min(height, top + crop_height), min(width, left + crop_width)
        if source_bottom > source_top and source_right > source_left:
            target_top, target_left = source_top - top, source_left - left
            output[
                :, target_top : target_top + source_bottom - source_top,
                target_left : target_left + source_right - source_left,
            ] = frames[:, source_top:source_bottom, source_left:source_right]
        return output

    def _augment(self, images):
        ops = (
            (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (adjust_contrast, random.uniform(0.5, 1.5)),
            (adjust_saturation, random.uniform(0.5, 1.5)),
            (adjust_hue, random.uniform(0.0, 0.0)),
        )
        flags = [random.randint(0, 1) == 0 for _ in ops]

        out = []
        for image in images:
            if self.swap_red_blue:
                image = Image.fromarray(np.asarray(image)[..., ::-1].copy())
            image = resize_image(image, factor=32, max_pixels=AUGMENT_MAX_PIXELS)
            for use, (op, value) in zip(flags, ops):
                if use:
                    image = op(image, value)
            out.append(image)
        return out

    @staticmethod
    def _messages(conversations, images):
        pool = [{"type": "image", "image": image} for image in images]
        messages = []

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                messages.append({"role": role, "content": [{"type": "text", "text": turn["value"]}]})
                continue

            content = []
            for part in PROMPT_RE.split(turn["value"]):
                if part == "<image>":
                    if not pool:
                        raise ValueError("number of <image> placeholders exceeds the number of provided images")
                    content.append(pool.pop(0))
                elif part == "<video>":
                    raise ValueError("video placeholders are not supported in the JSON dataset")
                elif part:
                    content.append({"type": "text", "text": part})
            messages.append({"role": role, "content": content})

        if pool:
            raise ValueError(f"{len(pool)} image(s) remain unused")
        return messages

    def _mask(self, traj, steps):
        kind = traj.get("trajectory_type", "success")
        if kind not in {"success", "ongoing", "invalid"}:
            raise ValueError(f"Unsupported trajectory_type: {kind}")
        mask = np.zeros(self.action_length, dtype=np.int32)
        mask[:steps] = 1
        return mask

    def _arm_action(self, traj, arm, frame, steps):
        rotm = self._frame(traj, f"proprios.{arm}_ee_rotm", frame).reshape(3, 3)
        pos = self._frame(traj, f"proprios.{arm}_ee_pos", frame)
        target_pos = self._future(traj, f"actions.{arm}_ee_pos", frame, steps)
        target_rotm = self._future(traj, f"actions.{arm}_ee_rotm", frame, steps).reshape(-1, 3, 3)
        return (
            self._pad((rotm.T @ (target_pos - pos).T).T, steps),
            self._pad(rotm2aa_batch(rotm.T @ target_rotm), steps),
            self._delta(traj, f"proprios.{arm}_gripper_pos", f"actions.{arm}_gripper_pos", frame, steps),
        )

    def _delta(self, traj, current_key, target_key, frame, steps):
        return self._pad(self._future(traj, target_key, frame, steps) - self._frame(traj, current_key, frame), steps)

    @staticmethod
    def _frame(traj, key, frame):
        return np.asarray(get_value(traj, key)[frame], dtype=np.float32)

    @staticmethod
    def _future(traj, key, frame, steps):
        return np.asarray(get_value(traj, key)[frame : frame + steps], dtype=np.float32)

    def _pad(self, value, steps):
        value = np.asarray(value)
        if steps == self.action_length:
            return value
        return np.concatenate([value, np.repeat(value[-1:], self.action_length - steps, axis=0)], axis=0)

    @staticmethod
    def _state(traj, frame):
        return compose_state(
            left_gripper=np.asarray(get_value(traj, "proprios.left_gripper_pos")[frame], dtype=np.float32),
            left_joint=np.asarray(get_value(traj, "proprios.left_arm_joint")[frame], dtype=np.float32),
            right_gripper=np.asarray(get_value(traj, "proprios.right_gripper_pos")[frame], dtype=np.float32),
            right_joint=np.asarray(get_value(traj, "proprios.right_arm_joint")[frame], dtype=np.float32),
        )
