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

EXP_DIR="oracle-$DATE"
ALPHA=.5
NUM_CLASSIFIERS=5
NUM_POINTS=200
NOISE_FUNC="gradientDescent"
ITERS=1000
LOG_FILE="oracle.log"

CMD="python -m binary_experiments -classes 4 9 -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_dir $EXP_DIR -alpha $ALPHA -num_point $NUM_POINTS -log_file $LOG_FILE"
eval $CMD.sh