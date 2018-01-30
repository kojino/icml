#!/bin/bash
#SBATCH -t 1-0:00
#SBATCH -p seas_dgx1
#SBATCH --mem=10000
#SBATCH -o sbatch_dl.out
#SBATCH -e sbatch_dl.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu


ALPHA=4
NOISE_TYPE="untargeted"
MWU_ITERS=100
LR=.001
OPT_ITERS=5000
DATA_PATH="imagenet_data"

CMD="python -m deep_learning_experiments -data_path $DATA_PATH -mwu_iters $MWU_ITERS -learning_rate $LR -noise_type $NOISE_TYPE -alpha $ALPHA -opt_iters $OPT_ITERS"
eval $CMD