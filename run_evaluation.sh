#!/bin/bash
#SBATCH --job-name=phase3_eval
#SBATCH --output=logsS/eval_%j.out
#SBATCH --error=logsS/eval_%j.err
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5:00:00

export HF_HOME="/home/morg/NLP_2526b/shairotman/hf_cache"
export TRANSFORMERS_CACHE="/home/morg/NLP_2526b/shairotman/hf_cache"

# הוספת הנתיב החדש של abc2midi שחילצנו
export PATH="/home/morg/NLP_2526b/shairotman/tools_env/bin:/home/morg/NLP_2526b/shairotman/NLP_final/abcmidi_local/usr/bin:$PATH"

cd /home/morg/NLP_2526b/shairotman/NLP_final
mkdir -p logsS

# הפעלת הסביבה המקומית
source /home/morg/NLP_2526b/shairotman/nlp_env/bin/activate
# Define paths and evaluation parameters
JSON_INPUT="zero_shot_baseline_results.json"
#JSON_INPUT="finetuned_model_results.json"
TARGET_COL="ground_truth_completion"
#TARGET_COL="finetuned_generated"

# הרצת הפייפליין
python evaluate.py 