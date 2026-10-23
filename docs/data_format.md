# Training data format

A training set is one directory with three parallel trees:

```
<root>/data/chunk-XXXX/episode_XXXXXXX.json                         # one annotation per episode
<root>/videos/chunk-XXXX/episode_XXXXXXX.{ego,wrist_left,wrist_right}.mp4
<root>/thumbs/chunk-XXXX/episode_XXXXXXX.ego<RES>s<STRIDE>.npy       # history thumbnails
```

`data.params.train_datasets.paths` points at `<root>/data`. The thumbnail path is derived from the
JSON path by replacing `/data/` with `/thumbs/`.

## Episode JSON

```json
{
  "num_frames": 1000,
  "instruction": {
    "general": [
      {
        "images": ["observations.ego", "observations.wrist_left", "observations.wrist_right"],
        "conversations": [
          {"from": "human", "value": "... # Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n<task instruction>"},
          {"from": "gpt", "value": ""}
        ]
      }
    ]
  },
  "observations": {
    "ego":         [{"path": "<root>/videos/.../episode_XXXXXXX.ego.mp4", "start": 0, "crop_bbox": null}],
    "wrist_left":  [{"path": "...", "start": 0, "crop_bbox": null}],
    "wrist_right": [{"path": "...", "start": 0, "crop_bbox": null}]
  },
  "proprios": {
    "left_ee_pos": [], "left_ee_rotm": [], "left_arm_joint": [], "left_gripper_pos": [],
    "right_ee_pos": [], "right_ee_rotm": [], "right_arm_joint": [], "right_gripper_pos": [],
    "waist_pos": []
  },
  "actions": {
    "left_ee_pos": [], "left_ee_rotm": [], "left_gripper_pos": [],
    "right_ee_pos": [], "right_ee_rotm": [], "right_gripper_pos": [],
    "waist_pos": [], "base_vel": []
  }
}
```

- Every array under `proprios` and `actions` has one entry per frame. Rotation matrices are flattened
  to 9 values; arm joints have up to 7 values.
- Each `<image>` placeholder takes the next entry of `images`. The empty assistant turn is filled by
  the data loader.
- A video entry needs `path`; `start` defaults to 0. `crop_bbox` is `null` or
  `(top, left, height, width)`; crops outside the frame are zero-padded.
- Images and states of one frame are aligned (`observation_image_lag: 0` in the data config).

## Model inputs and targets

Per sample at frame `t`:

- current views: ego, left wrist, right wrist;
- history: up to `history_frames` (20) ego thumbnails at `t - history_stride*k` (stride 25), read
  from the `.npy` cache at `floor(anchor / thumbs_stride)` and pooled to 16 tokens each;
- state: packed 60-dim vector (arm joints and grippers), quantile-normalized with `q01`/`q99`;
- target: a 30-step chunk of relative actions, packed into `(30, 60)` and normalized per step with
  `mean`/`std`:

| Dimensions | Value |
|---|---|
| 0-2 | left end-effector position |
| 3-5 | left end-effector axis-angle rotation |
| 6 | left gripper |
| 8-10 | right end-effector position |
| 11-13 | right end-effector axis-angle rotation |
| 14 | right gripper |
| 16 | waist |
| 17-19 | base velocity |
| other | reserved |

`configs/data/q2.yaml` carries the statistics of the RoboDojo training set; `tools/compute_normalize.py`
computes them for another set.
