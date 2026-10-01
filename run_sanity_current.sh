#!/bin/bash
#SBATCH --job-name=sanity_current
#SBATCH --output=sanity_current_%j.out
#SBATCH --error=sanity_current_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=s-004
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/morg/NLP_2526b/shairotman/nlp_env/bin/activate

cd /home/morg/NLP_2526b/shairotman/NLP_final
python -u sanity_overfit_current.py
