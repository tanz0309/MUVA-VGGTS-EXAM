#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
action="${1:-help}"
model="${2:-}"
if [[ "$action" == "prepare" ]]; then
  exec /opt/conda/envs/insid3-muva/bin/python "$TASK_ROOT/scripts/prepare_data.py"
fi
if [[ "$action" == "help" || -z "$model" ]]; then
  echo 'Usage: bash run.sh {smoke|evaluate|smoke-train|train} {insid3|vggts|omama|v2sam|v2sam-visual|v2sam-anchor|v2sam-multi} [options]'
  echo 'VGGT-S now uses the pinned official-code MUVA entry. Old checkpoints are incompatible.'
  echo 'VGGT-S generates missing points on GPU during train/evaluate and reuses its cache.'
  echo 'Optional VGGT-S precompute: GPU_ID=0 bash run.sh prepare-points vggts --split all'
  echo 'Optional cache inspection: bash run.sh check-data vggts --split all'
  echo 'Data preparation: bash run.sh prepare'
  exit 0
fi
runner="$TASK_ROOT/scripts/run_experiment.py"
model_args=(--model "$model")
case "$model" in
  insid3) model_python=/opt/conda/envs/insid3-muva/bin/python ;;
  vggts|vggts-official)
    model_python=/opt/conda/envs/vggts/bin/python
    runner="$TASK_ROOT/scripts/run_vggts_official.py"
    model_args=()
    ;;
  omama) model_python=/opt/conda/envs/omama/bin/python ;;
  v2sam|v2sam-visual|v2sam-anchor|v2sam-multi) model_python=/personal_workspace/TanZhuzhu/.conda/envs/v2sam/bin/python ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
shift 2
export CONDA_PREFIX="${model_python%/bin/python}"
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ "$model" == v2sam* ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export YOLO_CONFIG_DIR="$TASK_ROOT/cache/ultralytics"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TASK_ROOT/cache" "$TASK_ROOT/logs"
cd "$TASK_ROOT"
task_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
task_log="$TASK_ROOT/logs/${model}_${action}_${task_stamp}.log"
"$model_python" -u "$runner" "${model_args[@]}" --mode "$action" "$@" 2>&1 | tee "$task_log"
