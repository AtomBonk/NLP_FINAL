#!/bin/bash
#SBATCH --job-name=gen_finetune
#SBATCH --output=gen_%j.out
#SBATCH --error=gen_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --nodelist=s-[004-006]
#SBATCH --time=20:00:00

# הפעלת הסביבה
source activate nlp_env

# מעבר מפורש לתיקיית הפרויקט
cd /home/morg/NLP_2526b/shairotman/NLP_final

# הרצת קוד ההפקה
python generate_finetuned_samples.py