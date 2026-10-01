#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --output=sanity_%j.out
#SBATCH --error=sanity_%j.err
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

cd /home/morg/NLP_2526b/hilaetziony

# Environment setup
export HF_HOME=/home/morg/NLP_2526b/hilaetziony/hf_cache
export TRANSFORMERS_CACHE=/home/morg/NLP_2526b/hilaetziony/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ENABLE_XET=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source nlp_env/bin/activate

python -u train_sanity_check.py