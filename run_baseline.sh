#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=zero_shot_%j.out
#SBATCH --error=zero_shot_%j.err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --nodelist=s-[004-006]
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=15:00:00

# הגדרות סביבה ובטיחות לפני הריצה
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="/home/morg/NLP_2526b/shairotman/.cache/huggingface"

# הפעלת הסביבה וריצה
source activate nlp_env
cd /home/morg/NLP_2526b/shairotman/NLP_final
python zero_shot_baseline.py