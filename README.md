# Rex-M1-preview

Training and inference code of Rex-M1-preview, a bimanual manipulation policy for the RoboDojo
simulation tasks (`arx_x5`, end-effector actions).

```
configs/    Hydra recipe: q2.yaml (default), data/q2.yaml, model/, trainer/
mibot/      model (Qwen3-VL backbone + action head), dataset, collator, inference server
tools/      train.py, precompute_thumbs.py, compute_normalize.py, weight_convert.py
scripts/    train.sh
adapter/    RoboDojo / XPolicyLab policy adapter (Rex_M1_preview)
docs/       data_format.md
```

## Installation

```bash
conda create -n rex python=3.10 -y && conda activate rex
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

Weights needed for training:

- the Xiaomi-Robotics-1-5B base model as a `model_states.pt` (`tools/weight_convert.py --model_path <Xiaomi-Robotics-1-5B> --output_dir <dir>` converts the Hugging Face checkpoint into this format);
- the `Qwen/Qwen3-VL-4B-Instruct` processor and backbone (Hugging Face id or a local path).

`scripts/train.sh` runs with `HF_HUB_OFFLINE=1`, so download them beforehand.

## Data preparation

1. Download the RoboDojo simulation HDF5 episodes (`data/RoboDojo`, robot `arx_x5`) from the official
   RoboDojo release.
2. Convert them to the JSON + video format with the RoboDojo XPolicyLab tool:

   ```bash
   cd RoboDojo/XPolicyLab/policy/Xiaomi_Robotics_1
   RAW_DATA_ROOT=/path/to/RoboDojo/data/RoboDojo \
   OUTPUT_DIR=/path/to/RoboDojo-all-arx_x5-ee-v2 \
   bash process_data.sh RoboDojo all arx_x5 ee
   ```

3. Precompute the head-camera history thumbnails:

   ```bash
   python tools/precompute_thumbs.py --data-root /path/to/RoboDojo-all-arx_x5-ee-v2 --res 400 --stride 5 --workers 16
   ```

The format is described in [docs/data_format.md](docs/data_format.md). The normalization statistics of
this training set are in `configs/data/q2.yaml`; for other data compute them with
`tools/compute_normalize.py`.

## Training

The default recipe is `configs/q2.yaml`:

| Setting | Value |
|---|---|
| steps | 60,000 |
| GPUs | 4 nodes x 8 GPUs, batch 8 per GPU |
| optimizer | FusedAdam, betas (0.9, 0.95), weight decay 0.1, grad clip 1.0 |
| learning rate | cosine, warmup 500 steps from 5e-7, peak 1e-5, min 5e-7 |
| precision | bf16, DeepSpeed |
| action chunk | 30 steps |
| history | 20 frames, stride 25, thumbnails 400 px pooled to 16 tokens, no frame dropout |

Run on every node, with the same arguments:

```bash
NODE_RANK=<0..3> MASTER_ADDR=<node-0 address> bash scripts/train.sh \
  model.params.pretrained=/path/to/base/model_states.pt \
  data.params.train_datasets.paths=[/path/to/RoboDojo-all-arx_x5-ee-v2/data] \
  trainer.default_root_dir=/path/to/runs/q2
```

Other useful commands:

```bash
bash scripts/train.sh --cfg job --resolve                      # print the resolved config
bash scripts/train.sh launch.nodes=1 launch.gpus=1 trainer.max_steps=1 \
  data.params.train_datasets.batch_size=1 ...                  # one-step check on one GPU
bash scripts/train.sh trainer.ckpt_path=/path/to/last.ckpt ... # resume
```

Logging uses Weights & Biases (`WANDB_MODE=offline` to log locally). The run directory is
`<trainer.default_root_dir>/project_rex/q2/` (`trainer.project`, `trainer.exp_name`); it holds the training
`config.py` and a checkpoint every 5,000 steps.

## Inference

A checkpoint directory (for example `<run>/epoch=0-step=60000.ckpt`) is loaded together with the
`config.py` of its run directory.

Socket server:

```bash
python -m mibot.server.deploy --model /path/to/runs/q2/project_rex/q2/epoch=0-step=60000.ckpt --port 10086
```

RoboDojo adapter: `adapter/` is an XPolicyLab policy (`policy_name: Rex_M1_preview`). Copy
`adapter/{__init__.py,deploy.py,deploy.yml,model.py}` into `RoboDojo/XPolicyLab/policy/Rex_M1_preview/`
(next to the launch scripts of `policy/Xiaomi_Robotics_1/`), install this repository in the policy
environment, set `model_dir` in `deploy.yml` to the checkpoint, and run the XPolicyLab evaluation as
for other policies. Adapter settings: `action_type: ee`, `action_length: 15` (executed steps per
30-step chunk), `dtype: bf16`, `num_steps: 5`, `obs_delay: auto` (taken from the checkpoint's training
config), `eval_batch: true`.

## Contributors

Zixuan Guo\*, Yiran Ling\*, Qing Lian†, Peng Yun, Yuda Xiong, Kent Yu, Lei Zhang

\* Equal contribution  † Project lead

## Acknowledgements

Rex-M1-preview is post-trained from [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1).
We thank the Xiaomi Robotics team for releasing the model and code.

## License

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
