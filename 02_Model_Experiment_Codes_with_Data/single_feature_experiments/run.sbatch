#!/bin/bash
#SBATCH --job-name=pulsebat_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-287
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

# Update these paths for the compute environment.
export DATA_ROOT="${DATA_ROOT:-/path/to/04_Model_Input_Feature_Tables}"
export RESULT_BASE_DIR="${RESULT_BASE_DIR:-$PWD}"

# The default public matrix uses 3 LFP materials and 96 chunks per material:
# SLURM_ARRAY_TASK_ID = material_index * PARTS_PER_MATERIAL + part_index.
export PARTS_PER_MATERIAL="${PARTS_PER_MATERIAL:-96}"

# Optional overrides:
# export MODELS="linear,ridge,lasso,en,svm,rf,xgb,gpr,mlp,transformer,informer"
# export SEEDS="0,1,2,3,4"
# export TORCH_EPOCHS=300

python experiment.py
