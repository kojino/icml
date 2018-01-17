#!/bin/bash
#
#SBATCH -t 4-0:00
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=3000
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

ALPHA=.5
EXP_TYPE="multiclass"
NUM_CLASSIFIERS=5
NUM_POINTS=100
NOISE_FUNC="oracle"
ITERS=500
DATA_PATH="multiclass_data"

CMD="python -m binary_experiments -data_path $DATA_PATH -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_type $EXP_TYPE -alpha $ALPHA"
eval $CMD