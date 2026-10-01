#!/bin/bash
#SBATCH --job-name=inference_test
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --nodelist=s-004,s-005
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# הפעלת הסביבה הווירטואלית
source activate nlp_env

# הרצת קוד יצירת התוצאות
python generate_test_results.py