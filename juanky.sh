#!/bin/bash
#
#SBATCH -t 2-0:00
#SBATCH -p seas_dgx1
#SBATCH --mem=8000
#SBATCH -o sbatch.out
#SBATCH -e sbatch.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu
DATE="$(date +%m-%d)"

EXP_DIR="untargeted_$DATE"
LOG_FILE="untargeted.log"
DATA_PATH="imagenet_data"
TARGETED=0

CMD="python dawn_song_comparison.py -exp_dir $EXP_DIR -log_file $LOG_FILE -targeted $TARGETED -data_path $DATA_PATH"
eval $CMD