#!/bin/bash
#
#SBATCH -t 0-5:00
#SBATCH -p general
#SBATCH --mem=2000
#SBATCH -o sbatch.out
#SBATCH -e sbatch.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu
DATE="$(date +%m-%d)"

EXP_DIR="random-$DATE"
ALPHA=.5
NUM_CLASSIFIERS=5
NUM_POINTS=200
NOISE_FUNC="randomAscent"
ITERS=5
LOG_FILE="oracle.log"
DATA_PATH="binary_data"

CMD="python -m binary_experiments -data_path $DATA_PATH -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_dir $EXP_DIR -alpha $ALPHA  -log_file $LOG_FILE"
eval $CMD