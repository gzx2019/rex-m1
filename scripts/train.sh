#!/usr/bin/env bash
# Usage (from the repository root):
#   bash scripts/train.sh                                   # Q2 recipe (configs/q2.yaml)
#   bash scripts/train.sh trainer.max_steps=1 launch.nodes=1 launch.gpus=1   # one-step check
#   bash scripts/train.sh --cfg job --resolve               # print the resolved config only
#
# Every other argument is passed to Hydra; a later value of the same key wins.
# Common overrides:
#   trainer.exp_name=NAME                         run name
#   trainer.max_steps=N / trainer.save_interval=N
#   trainer.ckpt_path=/path/to.ckpt               resume a full training state
#   trainer.default_root_dir=/path                checkpoint directory
#   data.params.train_datasets.batch_size=N       batch per GPU
#   data.params.train_datasets.paths=[/path]      training JSON directory
#   model.params.pretrained=/path/to.pt           initial weights
#   launch.nodes=N launch.gpus=N                  topology (multi-node: set NODE_RANK and MASTER_ADDR)
set -euo pipefail
REX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REX_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=${WANDB_MODE:-online}
unset WANDB_DISABLED

config_name=q2
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name)
      config_name=${2:?--config-name requires a value}
      shift 2
      ;;
    --config-name=*)
      config_name=${1#*=}
      [[ -n "$config_name" ]] || { echo "--config-name requires a value" >&2; exit 2; }
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

exec python -B tools/train.py --config-name "$config_name" "${args[@]}"
