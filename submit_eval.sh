#!/bin/bash

# קליטת הארגומנטים שתעביר לסקריפט
JSON_FILE=$1
GEN_KEY=$2
RUN_NAME=$3

# יצירת התיקייה המרכזית שתכיל את כל הפלטים
mkdir -p evaluation_outputs

echo "Submitting Slurm job for: $RUN_NAME (using $JSON_FILE)"

# יצירה ושליחה של קובץ Slurm דינמי (Heredoc)
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=evaluation_outputs/${RUN_NAME}_%j.out
#SBATCH --error=evaluation_outputs/${RUN_NAME}_%j.err
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# הגדרת נתיבי התוכנות (כדי ש-abc2midi יעבוד)
export PATH="/home/morg/NLP_2526b/shairotman/tools_env/bin:/home/morg/NLP_2526b/shairotman/NLP_final/abcmidi_local/usr/bin:\$PATH"

cd /home/morg/NLP_2526b/shairotman/NLP_final

# הפעלת הסביבה
source /home/morg/NLP_2526b/shairotman/nlp_env/bin/activate

# הרצת פייתון עם הארגומנטים שהועברו
python -u evaluate_all.py --results_file "$JSON_FILE" --eval_dir "evaluation_outputs/$RUN_NAME" --gen_key "$GEN_KEY"
EOT