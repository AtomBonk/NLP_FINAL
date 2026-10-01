#!/bin/bash
#SBATCH --job-name=full_finetune
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=s-[004-006]
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# כיבוי מנגנון הורדה שנוטה לקרוס
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# הפניה מפורשת לכונן הזמני והגדול של שרת החישוב!
export HF_HOME="/tmp/amit_hf_cache"
source activate nlp_env

cd /home/morg/NLP_2526b/shairotman/NLP_final
python finetune.py