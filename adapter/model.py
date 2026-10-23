# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
# Modified by the Rex-M1-preview authors (2026): long-context history buffering, prompt assembly and
# constant-budget token pooling on the inference path.

"""Rex_M1_preview policy for XPolicyLab evaluation.

This adapter drives the vendored ``xr1`` model in-process, reproducing exactly
what ``mibot/server/deploy.py`` + ``mibot/server/runtime/server.py`` +
``mibot/server/runtime/client.py`` do over a socket, minus the socket.

Model loading mirrors ``mibot.server.deploy``:
    cfg, model = load_model(model_dir, device)
    mean, std, q01, q99, action_mask = load_stats(cfg, device)

Batch assembly mirrors ``mibot.server.runtime.client.Client.__call__``:
  three PIL views (ego / left-wrist / right-wrist) resized with
  ``mibot.utils.io.resize_image`` and tokenized through the Qwen3-VL chat
  template with ``do_resize=False``, plus a packed 60-dim state.

Normalization mirrors ``Server.run``: the state is quantile-mapped to [-1, 1]
and clamped, the action chunk is produced in gaussian-normalized space and
denormalized with ``mean``/``std``, both masked by ``action_mask``.

Vector layouts (from ``mibot/utils/io.py``; note state and action differ):
  State - ``compose_state``, always joint space, shape (1, 60):
    [0:7] left_arm_joint, [7:8] left_gripper,
    [8:15] right_arm_joint, [15:16] right_gripper; every other slot is zero.
  Action - ``ACTION_PARTS``, shape (30, 60), all values are RELATIVE deltas:
    [0:3] left_ee_pos, [3:6] left_ee_aa, [6:7] left_gripper,
    [8:11] right_ee_pos, [11:14] right_ee_aa, [14:15] right_gripper,
    [16:17] waist, [17:20] base_vel; every other slot is padding.

Only ``action_type="ee"`` is supported: the packed action carries no arm-joint
slots, so joint targets cannot be recovered from the model output.
"""

from __future__ import annotations

from pathlib import Path
import os
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate

_SUPPORTED_BENCH_NAMES = ("RoboDojo", "RoboDojo_real")

# Per-bench EEF local axis redefinition matrices.
# Convention: R_mibot = R_env @ P; position and gripper unchanged.

# RoboDojo (sim): P = Rx(+90°) @ Rz(+90°), same for all robots.
_EEF_REFRAME_P_ROBODOJO = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

# RoboDojo_real: per-robot P matrices.
_EEF_REFRAME_P_ROBODOJO_REAL = {
    "piper_x": np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    ),
    "piper": np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    ),
    "arx_x5": np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    ),
}


def _get_eef_reframe_p(bench_name: str, env_cfg_type: str) -> np.ndarray:
    if bench_name == "RoboDojo":
        return _EEF_REFRAME_P_ROBODOJO
    elif bench_name == "RoboDojo_real":
        if env_cfg_type not in _EEF_REFRAME_P_ROBODOJO_REAL:
            raise ValueError(
                f"[Rex_M1_preview] Unsupported env_cfg_type={env_cfg_type!r} "
                f"for bench_name='RoboDojo_real'. "
                f"Supported: {list(_EEF_REFRAME_P_ROBODOJO_REAL.keys())}"
            )
        return _EEF_REFRAME_P_ROBODOJO_REAL[env_cfg_type]
    else:
        raise ValueError(
            f"[Rex_M1_preview] Unsupported bench_name={bench_name!r}. "
            f"Supported: {list(_SUPPORTED_BENCH_NAMES)}"
        )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Convert observation image to HWC uint8 RGB ndarray."""
    image = np.asarray(image)
    if image.ndim == 3:
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image(obs: dict, cam_keys: list[str]) -> np.ndarray:
    """Extract image from XPolicyLab observation dict."""
    vision = obs.get("vision", {})
    for key in cam_keys:
        if key not in vision:
            continue
        cam = vision[key]
        if isinstance(cam, dict):
            for img_key in ("color", "colors", "rgb"):
                if img_key in cam:
                    return _ensure_hwc_uint8(cam[img_key])
        else:
            return _ensure_hwc_uint8(cam)
    raise KeyError(f"No image found for camera keys: {cam_keys}")


def _ee_pose_sim_to_mibot(
    xyz_sim: np.ndarray, quat_wxyz_sim: np.ndarray, eef_reframe_p: np.ndarray
):
    """Convert an ee pose from environment frame to MiBot frame.

    Only the EEF local axes are redefined (R_mibot = R_env @ P); the base-frame
    position is unchanged. Returns (pos, rotm) as float64 for downstream math.
    """
    pos_m = np.asarray(xyz_sim, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz_sim, dtype=np.float64).reshape(4)
    rotm_sim = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    rotm_m = rotm_sim @ eef_reframe_p
    return pos_m, rotm_m


def _ee_pose_mibot_to_sim(
    pos_mibot: np.ndarray, rotm_mibot: np.ndarray, eef_reframe_p_inv: np.ndarray
):
    """Convert an ee pose from MiBot frame back to environment frame.

    Inverse of :func:`_ee_pose_sim_to_mibot`: position unchanged, rotation
    mapped by R_env = R_mibot @ P^T. Returns (xyz, quat_wxyz) as float32.
    """
    xyz = np.asarray(pos_mibot, dtype=np.float32).reshape(3)
    rotm_sim = np.asarray(rotm_mibot, dtype=np.float64) @ eef_reframe_p_inv
    quat_xyzw = Rotation.from_matrix(rotm_sim).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float64)
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz
    return xyz, quat_wxyz.astype(np.float32)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "ee")
        if self.action_type != "ee":
            raise ValueError(
                f"[Rex_M1_preview] Unsupported action_type: {self.action_type!r}. "
                "The packed 60-dim action of this model carries end-effector "
                "slots only (see ACTION_PARTS in mibot/utils/io.py), so joint "
                "targets cannot be recovered from its output. Set "
                "action_type='ee' in deploy.yml."
            )
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.bench_name = model_cfg["bench_name"]
        self._eef_reframe_p = _get_eef_reframe_p(self.bench_name, self.env_cfg_type)
        self._eef_reframe_p_inv = self._eef_reframe_p.T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.default_prompt = model_cfg.get(
            "default_prompt", model_cfg.get("task_name", "Perform the task.")
        )
        # Image preprocessing, matching mibot Client / JsonDataset._augment.
        self.image_factor = int(model_cfg.get("image_factor", 32))
        self.image_max_pixels = int(model_cfg.get("image_max_pixels", 160000))
        # Number of leading action steps actually executed per inference call;
        # 0 or None means the whole predicted chunk.
        self.action_length = model_cfg.get("action_length") or 0
        print(f"[Rex_M1_preview] action_length={self.action_length or 'whole chunk'} "
              f"(executed steps per inference; 0 = whole predicted chunk)", flush=True)

        import mibot
        xr1_root = Path(mibot.__file__).resolve().parents[1]
        print(f"[Rex_M1_preview] xr1 package root: {xr1_root}", flush=True)

        from mibot.server.deploy import load_model, load_stats
        from mibot.utils.io import ACTION_EPS, build_action_mask, compose_state, denormalize_action

        self._action_eps = ACTION_EPS
        self._compose_state = compose_state
        self._denormalize_action = denormalize_action
        self._build_action_mask = build_action_mask

        if not model_cfg.get("model_dir"):
            raise ValueError("model_dir must identify an explicit checkpoint or package")
        model_dir = str(Path(model_cfg["model_dir"]).expanduser().absolute())
        print(f"[Rex_M1_preview] Loading model from {model_dir}...", flush=True)
        cfg, self.model = load_model(model_dir, str(self.device),
            dtype=model_cfg.get("dtype", "bf16"), num_steps=model_cfg.get("num_steps", 5),
            backbone_path=model_cfg.get("vlm_processor_path"))
        self.cfg = cfg
        self.mean, self.std, self.q01, self.q99, self.action_mask = load_stats(
            cfg, str(self.device)
        )
        self.action_shape = tuple(self.mean.shape)
        self._state_valid = self.q99 > self.q01

        self.processor = self._build_processor(model_cfg)

        # --- Long-context history.
        self._configure_history(cfg, model_cfg)

        # Internal state
        self._lazy_images = bool(model_cfg.get("lazy_images", True))
        self._encoded_obs_list: list[dict[str, Any]] = []
        # Per-env ring buffers, keyed by obs["env_idx"].
        self._hist_by_env: dict[Any, deque] = {}
        self._initial_by_env: dict[Any, Image.Image] = {}
        # Training release owns image/state timing: legacy lag1, shift-fixed v2 lag0.
        from mibot.observation_contract import observation_contract
        self.observation_contract = observation_contract(cfg.data.params.train_datasets, model_cfg.get("obs_delay", "auto"))
        self._obs_delay = bool(self.observation_contract["image_lag"])
        print(f"[Rex_M1_preview] training_image_lag={int(self._obs_delay)}", flush=True)
        # Each environment has an optional per-episode CPU noise generator.
        _ns = model_cfg.get("noise_seed")
        self._noise_seed = int(_ns) if _ns not in (None, "") else None
        self._noise_gens = {}
        print(f"[Rex_M1_preview] noise_seed={self._noise_seed}", flush=True)
        # Precision is passed explicitly from the deployment configuration.
        self._rex_dtype = torch.float32 if model_cfg.get("dtype", "bf16") == "fp32" else torch.bfloat16
        # Optional inference state traces.
        self._state_log = None
        if model_cfg.get("state_log"):
            os.makedirs(model_cfg["state_log"], exist_ok=True)
            self._state_log = open(os.path.join(model_cfg["state_log"], f"pid{os.getpid()}.jsonl"), "a", buffering=1)
        self._rex_episode = 0
        # Optional inference batch dumps.
        self._dump_dir = model_cfg.get("dump_batch") or None
        self._dump_n = 0
        if self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)
        print(f"[Rex_M1_preview] dtype={self._rex_dtype} state_log={bool(self._state_log)} dump={self._dump_dir}", flush=True)
        self._prev_imgs_by_env: dict[Any, tuple] = {}
        print(f"[Rex_M1_preview] obs_delay={'ON (images lag state by one step)' if self._obs_delay else 'off'}", flush=True)

        print(
            f"[Rex_M1_preview] Model loaded. action_shape={self.action_shape}, "
            f"action_type={self.action_type}, device={self.device}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _configure_history(self, cfg, model_cfg: dict[str, Any]) -> None:
        # Checkpoint values take precedence; deployment defaults fill only missing keys.
        data = {**model_cfg, **cfg.data.params.train_datasets}
        self.history_frames = int(data.get("history_frames", 0) or 0)
        self.history_stride = int(data.get("history_stride", 25))
        self.history_res = int(data.get("history_res", 128))
        self.history_pool = int(data.get("history_pool", 0) or 0)
        self.history_dropout = int(data.get("history_dropout", 1))
        # This is a training contract, never a deployment override.
        self.initial_frame = cfg.data.params.train_datasets.get("initial_frame", 0)
        if self.initial_frame not in (0, 1):
            raise ValueError("initial_frame must be 0 or 1")
        if self.initial_frame and (self.history_frames <= 0 or self.history_pool != 16):
            raise ValueError("initial_frame requires history_frames > 0 and history_pool=16")
        if self.history_stride <= 0:
            raise ValueError("history_stride must be positive")
        print(f"[Rex_M1_preview] initial_frame={int(self.initial_frame)}", flush=True)
        if self.history_pool and self.history_pool != 16:
            raise ValueError(
                f"[Rex_M1_preview] unsupported history_pool={self.history_pool} (only 0 or 16)"
            )
        ptp_steps = data.get("ptp_steps", [0, 5, 10, 15, 20])
        self.ptp_steps = [int(v) for v in ptp_steps]
        self.ptp_length = int(getattr(self.model, "ptp_length", 0) or 0)
        expected = self.history_frames * len(self.ptp_steps)
        if self.ptp_length not in (0, expected):
            raise ValueError(
                f"[Rex_M1_preview] checkpoint ptp_length={self.ptp_length} but history config implies "
                f"{expected} (history_frames={self.history_frames} x {len(self.ptp_steps)} steps)"
            )
        if self.history_frames:
            print(
                f"[Rex_M1_preview] history enabled: N_max={self.history_frames} stride={self.history_stride} "
                f"res={self.history_res} pool={self.history_pool} dropout={self.history_dropout} "
                f"ptp_length={self.ptp_length}",
                flush=True,
            )

    @staticmethod
    def _build_processor(model_cfg: dict[str, Any]):
        from mibot.utils.processor import DEFAULT_PROCESSOR, build_processor
        return build_processor(model_cfg.get("vlm_processor_path", DEFAULT_PROCESSOR))

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs):
        """Chat turns, byte-for-byte identical to mibot Client._messages."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following observations are captured from multiple views.\n# Ego View\n",
                    },
                    {"type": "image", "image": ego_obs},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left_wrist_obs},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": right_wrist_obs},
                    {
                        "type": "text",
                        "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    @staticmethod
    def _messages_with_history(instruction, hist_images, ego_obs, left_wrist_obs, right_wrist_obs,
                               initial_image=None):
        """History images first, then the original user turn unchanged."""
        msgs = Model._messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs)
        instr = str(instruction).replace("/no_cot", "").strip()
        prefix = f"Task: {instr}\n"
        if initial_image is None:
            head = [{"type": "text", "text": prefix + f"# History (oldest first, {len(hist_images)} frames)\n"}]
        else:
            head = [{"type": "text", "text": prefix + "# Initial Ego View (episode start)\n"},
                    {"type": "image", "image": initial_image},
                    {"type": "text", "text": f"\n# History (oldest first, {len(hist_images)} frames)\n"}]
        head += [{"type": "image", "image": im} for im in hist_images]
        if hist_images:
            head.append({"type": "text", "text": "\n"})
        msgs[0]["content"] = head + msgs[0]["content"]
        return msgs

    def _push_history(self, env_key, head_img: np.ndarray) -> None:
        from mibot.utils.io import resize_image
        buf = self._hist_by_env.get(env_key)
        if buf is None:
            buf = deque(maxlen=self.history_stride * self.history_frames + 1)  # +1: current frame included
            self._hist_by_env[env_key] = buf
        image = resize_image(Image.fromarray(head_img), factor=self.image_factor,
                             max_pixels=self.history_res * self.history_res)
        buf.append(image)
        if self.initial_frame and env_key not in self._initial_by_env:
            self._initial_by_env[env_key] = image.copy()

    def _history_for(self, env_key):
        """Anchors t - stride*k, k = N..1 (oldest first), N = min(N_max, floor((len-1)/stride))."""
        buf = self._hist_by_env.get(env_key)
        if not buf:
            return []
        last = len(buf) - 1
        n = min(self.history_frames, last // self.history_stride)
        return [buf[last - self.history_stride * k] for k in range(n, 0, -1)]

    def _encode_observation(self, obs: dict[str, Any], defer_images: bool = False) -> dict[str, Any]:
        """Encode a single XPolicyLab obs into the model's input representation.

        Returns the tokenized payload plus the absolute current pose needed to
        turn the model's relative action chunk back into absolute targets.
        """
        from mibot.utils.io import resize_image

        head_img = _extract_image(obs, ["cam_head", "cam_high", "head_camera"])
        left_img = _extract_image(obs, ["cam_left_wrist", "left_camera", "wrist_left"])
        right_img = _extract_image(
            obs, ["cam_right_wrist", "right_camera", "wrist_right"]
        )
        if self._obs_delay:
            _dk = obs.get("env_idx", obs.get("env_id", 0))
            _cur = (head_img.copy(), left_img.copy(), right_img.copy())
            _prev = self._prev_imgs_by_env.get(_dk)
            self._prev_imgs_by_env[_dk] = _cur
            if _prev is not None:
                head_img, left_img, right_img = _prev

        # Same preprocessing as mibot Client.__call__ and JsonDataset._augment:
        # resize to a factor-of-32 grid under max_pixels, then hand the images
        # to the processor with do_resize disabled.
        pil_images = [Image.fromarray(img) for img in (head_img, left_img, right_img)]
        if defer_images:
            # Validate geometry at update time, preserving errors even on skipped frames.
            from mibot.utils.io import resize_dimensions
            for img in pil_images:
                resize_dimensions(img.size, factor=self.image_factor, max_pixels=self.image_max_pixels)
        else:
            pil_images = [resize_image(img, factor=self.image_factor,
                                       max_pixels=self.image_max_pixels) for img in pil_images]

        state = obs.get("state", {})
        left_arm_joint = np.asarray(
            state["left_arm_joint_state"], dtype=np.float32
        ).reshape(-1)
        left_gripper = np.asarray(
            state["left_ee_joint_state"], dtype=np.float32
        ).reshape(-1)[:1]
        right_arm_joint = np.asarray(
            state["right_arm_joint_state"], dtype=np.float32
        ).reshape(-1)
        right_gripper = np.asarray(
            state["right_ee_joint_state"], dtype=np.float32
        ).reshape(-1)[:1]

        # compose_state packs joints into [0:7]/[8:15] and grippers into
        # [7:8]/[15:16], and rejects arms wider than 7 dof.
        state_np = self._compose_state(
            left_gripper=left_gripper,
            left_joint=left_arm_joint,
            right_gripper=right_gripper,
            right_joint=right_arm_joint,
        )  # (1, 60)

        # The model predicts RELATIVE deltas w.r.t. the observed pose, expressed
        # in the current ee frame (MiBot convention). Stash the absolute pose so
        # _actions_to_xpl_format can restore absolute targets.
        left_pose = np.asarray(state["left_ee_pose"], dtype=np.float64).reshape(7)
        right_pose = np.asarray(state["right_ee_pose"], dtype=np.float64).reshape(7)
        l_pos_m, l_rotm_m = _ee_pose_sim_to_mibot(
            left_pose[:3], left_pose[3:7], self._eef_reframe_p
        )
        r_pos_m, r_rotm_m = _ee_pose_sim_to_mibot(
            right_pose[:3], right_pose[3:7], self._eef_reframe_p
        )
        current_state = {
            "left_ee_pos_mibot": l_pos_m,
            "left_ee_rotm_mibot": l_rotm_m,
            "left_gripper": float(left_gripper[0]),
            "right_ee_pos_mibot": r_pos_m,
            "right_ee_rotm_mibot": r_rotm_m,
            "right_gripper": float(right_gripper[0]),
        }

        instruction = self._get_instruction(obs)

        # Vision + instruction turns only, exactly as mibot Client._messages.
        # The training-time "Robot state: <state>" / "<a_i>...<score>" turns are
        # deliberately NOT added: those tokens are embedded by the VLM only when
        # xr1.forward is in training mode (it passes state_embeds and reads the
        # <a_i>/<score> hidden states for the choice head). At inference the VLM
        # is called without state_embeds, so a <state> token would raise
        # "State tokens require state_embeds". The proprioception reaches the
        # model through batch["state"] -> DiT state_projector instead.
        n_hist = 0
        if self.history_frames:
            env_key = obs.get("env_idx", obs.get("env_id", 0))
            self._push_history(env_key, head_img)
            hist_images = self._history_for(env_key)
            n_hist = len(hist_images)
            initial_image = self._initial_by_env[env_key] if self.initial_frame else None
            messages = self._messages_with_history(instruction, hist_images, *pil_images,
                                                   initial_image=initial_image)
        else:
            messages = self._messages(instruction, *pil_images)

        return {
            "_pending_views": pil_images if defer_images else None,
            "messages": messages,
            "state": torch.from_numpy(state_np)[None],  # (1, 1, 60)
            "current_state": current_state,
            "n_hist": n_hist,
            **({"n_memory": n_hist + 1} if self.initial_frame else {}),
            "_rex_env": obs.get("env_idx", obs.get("env_id", 0)),
            "_rex_raw_state": ({k: state[k] for k in ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state") if k in state}
                               if self._state_log is not None else None),
        }

    def _get_instruction(self, obs: dict[str, Any]) -> str:
        for key in ("instruction", "instructions"):
            if key not in obs:
                continue
            val = obs[key]
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val.strip():
                return val.strip().rstrip(".") + "."
        return self.default_prompt

    def _apply_history_pooling(self, batch: dict[str, Any], n_hists: list[int]) -> dict[str, Any]:
        """Shrink the image-placeholder span of each sample's first ``n_hist`` history images to
        ``history_pool`` tokens and emit the matching ``image_pool_mask``. Position encodings are
        left to qwen3vl.get_rope_index; ``pixel_values`` and ``image_grid_thw`` are unchanged.
        """
        ids, attn, grids = batch["input_ids"], batch["attention_mask"], batch["image_grid_thw"]
        image_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        pad_id = self.processor.tokenizer.pad_token_id
        left_pad = getattr(self.processor.tokenizer, "padding_side", "right") == "left"

        rows: list[list[int]] = []
        pool_flags: list[bool] = []
        for index, n_hist in enumerate(n_hists):
            real = ids[index][attn[index] == 1].tolist()
            out: list[int] = []
            cursor, image_index = 0, 0
            while cursor < len(real):
                if real[cursor] != image_id:
                    out.append(real[cursor])
                    cursor += 1
                    continue
                end = cursor
                while end < len(real) and real[end] == image_id:
                    end += 1
                pooled = image_index < n_hist
                out.extend([image_id] * (self.history_pool if pooled else end - cursor))
                pool_flags.append(pooled)
                image_index += 1
                cursor = end
            rows.append(out)

        if len(pool_flags) != int(grids.shape[0]):
            raise ValueError(
                f"[Rex_M1_preview] counted {len(pool_flags)} images but image_grid_thw has "
                f"{int(grids.shape[0])} rows — the mask would be misaligned with the grid"
            )

        width = max(len(r) for r in rows)
        new_ids = ids.new_full((len(rows), width), pad_id)
        new_attn = torch.zeros_like(new_ids)
        for index, row in enumerate(rows):
            span = slice(width - len(row), width) if left_pad else slice(0, len(row))
            new_ids[index, span] = torch.tensor(row, dtype=ids.dtype, device=ids.device)
            new_attn[index, span] = 1
        batch["input_ids"] = new_ids
        batch["attention_mask"] = new_attn
        batch["image_pool_mask"] = torch.tensor(pool_flags, dtype=torch.bool, device=ids.device)
        return batch

    def _past_mask(self, n_hist: int) -> torch.Tensor:
        """(ptp_length, 60) mask for the past segment. Only reached when ptp_length > 0."""
        k = len(self.ptp_steps)
        temporal = np.zeros(self.ptp_length, dtype=np.int32)
        if n_hist:
            temporal[(self.history_frames - n_hist) * k :] = 1
        return torch.from_numpy(self._build_action_mask(self.ptp_length, temporal))

    def _build_batch(self, encoded_obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize a list of encoded observations into one model batch.

        Keys are exactly those ``mibot.server.runtime.client`` puts on the wire
        (``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
        ``state``), because ``xr1.forward`` forwards every leftover key straight
        into the VLM. In particular ``action_vlm_condition_segments`` is NOT set:
        ``xr1._unpad`` slices the cache as ``keys[:, 0, ...]``, which is only
        valid for the packed single-sequence batch used during training. With
        segments left unset the DiT attends over the full VLM cache, matching the
        official inference server.
        """
        batch = self.processor.apply_chat_template(
            [item["messages"] for item in encoded_obs_list],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        batch = dict(batch)
        if self.history_pool:
            if any("n_hist" not in item for item in encoded_obs_list):
                raise ValueError(
                    "[Rex_M1_preview] history_pool is enabled but an observation carries no n_hist — "
                    "this would silently fall back to no pooling, making inference disagree "
                    "with training"
                )
            batch = self._apply_history_pooling(
                batch, [int(item.get("n_memory", item["n_hist"])) for item in encoded_obs_list]
            )
        batch["state"] = torch.cat([item["state"] for item in encoded_obs_list], dim=0)
        if self.ptp_length:
            batch["past_action_mask"] = torch.stack([self._past_mask(item.get("n_hist", 0)) for item in encoded_obs_list])
        return batch

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference_batch(self, batch: dict[str, Any]) -> np.ndarray:
        """Normalize, run the model, denormalize. Mirrors ``Server.run``.

        Returns unnormalized relative actions as [B, action_length, 60].
        """
        batch = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        batch_size = batch["input_ids"].shape[0]

        # (B, action_length, 60) mask marking the slots the model was trained on.
        mask = self.action_mask.unsqueeze(0).expand(batch_size, -1, -1)
        batch["action_mask"] = mask
        # No action prefix at inference time: the chunk starts from pure noise.
        batch["action"] = torch.zeros(
            (batch_size, *self.action_shape),
            device=self.device,
            dtype=self._rex_dtype,
        )
        if self.ptp_length:
            batch["past_action"] = torch.zeros(
                (batch_size, self.ptp_length, self.action_shape[-1]), device=self.device, dtype=self._rex_dtype
            )
            batch["past_action_mask"] = batch["past_action_mask"].to(self.device)

        # Quantile-normalize the state to [-1, 1], leaving padded dims at zero.
        state = batch["state"]
        valid = self._state_valid
        normalized_state = torch.zeros_like(state)
        normalized_state[..., valid[0]] = (
            2.0
            * (state[..., valid[0]] - self.q01[..., valid[0]])
            / (self.q99[..., valid[0]] - self.q01[..., valid[0]] + self._action_eps)
            - 1.0
        )
        batch["state"] = normalized_state.clamp(-1.0, 1.0)

        action = self.model.generate(batch)
        if self.ptp_length:
            # generate() returns [past | future]; keep the future chunk before denormalizing.
            action = action[:, self.ptp_length :]
        action = self._denormalize_action(action * mask, self.mean, self.std) * mask
        return action.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Action postprocessing
    # ------------------------------------------------------------------

    def _actions_to_xpl_format(
        self, raw_actions: np.ndarray, current_state: dict[str, Any]
    ) -> list[dict[str, np.ndarray]]:
        """Convert one relative action chunk [T, 60] into XPolicyLab actions.

        Reproduces ``mibot.utils.io.recover_action`` in the MiBot frame and then
        maps the result back to the environment frame. Every packed slot is a
        delta w.r.t. the observed pose (see ACTION_PARTS):
            abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
            abs_rotm_m = current_rotm_m @ Rot(delta_axis_angle)
            abs_grip   = current_grip + delta_grip
        Slots [0:3]/[3:6]/[6:7] are the left arm, [8:11]/[11:14]/[14:15] the
        right; waist [16:17] and base [17:20] have no XPolicyLab counterpart on
        the supported robots and are ignored.
        """
        action_list = []
        for t in range(raw_actions.shape[0]):
            a = raw_actions[t]

            left_xyz, left_quat = self._restore_abs_ee(
                a[0:3], a[3:6],
                current_state["left_ee_pos_mibot"],
                current_state["left_ee_rotm_mibot"],
            )
            right_xyz, right_quat = self._restore_abs_ee(
                a[8:11], a[11:14],
                current_state["right_ee_pos_mibot"],
                current_state["right_ee_rotm_mibot"],
            )
            left_grip = current_state["left_gripper"] + float(a[6])
            right_grip = current_state["right_gripper"] + float(a[14])

            action_list.append({
                "left_ee_pose": np.concatenate([left_xyz, left_quat]).astype(np.float32),
                "right_ee_pose": np.concatenate([right_xyz, right_quat]).astype(np.float32),
                "left_ee_joint_state": np.array([left_grip], dtype=np.float32),
                "right_ee_joint_state": np.array([right_grip], dtype=np.float32),
            })

        return action_list

    def _restore_abs_ee(
        self,
        delta_pos: np.ndarray,
        delta_aa: np.ndarray,
        current_pos_m: np.ndarray,
        current_rotm_m: np.ndarray,
    ):
        """Restore an absolute ee pose (env frame) from an ee-frame delta.

        Inverse of the packing in ``JsonDataset._arm_action``, all in the MiBot
        frame, then converted back to the environment frame:
            abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
            abs_rotm_m = current_rotm_m @ Rot(delta_aa)
        Returns (xyz, quat_wxyz) in the environment frame.
        """
        delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape(3)
        abs_pos_m = current_pos_m + current_rotm_m @ delta_pos
        delta_rotm = Rotation.from_rotvec(
            np.asarray(delta_aa, dtype=np.float64).reshape(3)
        ).as_matrix()
        abs_rotm_m = current_rotm_m @ delta_rotm
        return _ee_pose_mibot_to_sim(abs_pos_m, abs_rotm_m, self._eef_reframe_p_inv)

    # ------------------------------------------------------------------
    # ModelTemplate interface
    # ------------------------------------------------------------------

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._encoded_obs_list = [self._encode_observation(obs, defer_images=self._lazy_images)
                                  for obs in obs_list]

    def get_action(self, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Rex_M1_preview] Call update_obs before get_action."
            )
        return self._predict_action_chunks(self._encoded_obs_list[:1])[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Rex_M1_preview] Call update_obs_batch before get_action_batch."
            )
        return self._predict_action_chunks(self._encoded_obs_list)

    def _materialize_images(self, encoded):
        """Resize current views once; history already advanced on every observation."""
        pending = encoded.get("_pending_views")
        if pending is None:
            return
        from mibot.utils.io import resize_image
        replacements = {id(img): resize_image(img, factor=self.image_factor,
                                              max_pixels=self.image_max_pixels)
                        for img in pending}
        for message in encoded["messages"]:
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if id(img) in replacements:
                        item["image"] = replacements[id(img)]
        encoded["_pending_views"] = None

    def _predict_action_chunks(
        self, encoded_obs_list: list[dict[str, Any]]
    ) -> list[list[dict[str, np.ndarray]]]:
        """Run one batched inference pass and convert each chunk."""
        for encoded in encoded_obs_list:
            self._materialize_images(encoded)
        batch = self._build_batch(encoded_obs_list)
        if self._dump_dir is not None:
            import copy as _copy
            _pre = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else _copy.deepcopy(v)) for k, v in batch.items()}
        if self._noise_seed is not None:
            self.model._noise_gens = [self._noise_gens.setdefault(e.get("_rex_env"), torch.Generator().manual_seed(self._noise_seed))
                                      for e in encoded_obs_list]
        try:
            raw_actions = self._run_inference_batch(batch)
        finally:
            self.model._noise_gens = None
        if self._dump_dir is not None:
            import time as _time
            torch.save({"batch": _pre, "current_state": [e["current_state"] for e in encoded_obs_list],
                        "env": [e.get("_rex_env") for e in encoded_obs_list], "episode": self._rex_episode,
                        "n": self._dump_n, "t": _time.time(), "live_raw": raw_actions, "dtype": str(self._rex_dtype)},
                       os.path.join(self._dump_dir, f"pid{os.getpid()}_ep{self._rex_episode:03d}_n{self._dump_n:05d}.pt"))
            self._dump_n += 1
        chunks = []
        for index, encoded_obs in enumerate(encoded_obs_list):
            actions = raw_actions[index]
            if 0 < self.action_length < actions.shape[0]:
                actions = actions[: self.action_length]
            chunks.append(
                self._actions_to_xpl_format(actions, encoded_obs["current_state"])
            )
        if self._state_log is not None:
            import json as _json, time as _time
            for encoded_obs, chunk in zip(encoded_obs_list, chunks):
                raw = encoded_obs.get("_rex_raw_state") or {}
                self._state_log.write(_json.dumps({
                    "t": _time.time(), "episode": self._rex_episode, "env": encoded_obs.get("_rex_env"),
                    "state": {k: np.asarray(v, dtype=np.float64).ravel().tolist() for k, v in raw.items()},
                    "cmd": [[*np.asarray(a["left_ee_pose"])[:3].tolist(), *np.asarray(a["right_ee_pose"])[:3].tolist(),
                             float(np.asarray(a["left_ee_joint_state"]).ravel()[0]), float(np.asarray(a["right_ee_joint_state"]).ravel()[0])]
                            for a in chunk],
                }) + "\n")
        return chunks

    def reset(self):
        self._rex_episode = getattr(self, "_rex_episode", 0) + 1
        self._encoded_obs_list = []
        self._hist_by_env = {}
        self._initial_by_env = {}
        self._prev_imgs_by_env = {}
        self._noise_gens = {}
        print("[Rex_M1_preview] Model reset.", flush=True)
