#!/bin/bash
#
#SBATCH -t 6-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

NOISE_TYPE="untargeted"
ALPHA=3.5
DATA_SET="imagenet"
MWU_ITERS=5
OPT_ITERS=1000
LR=.1

CMD="python -m deep_learning_experiments -data_set $DATA_SET -mwu_iters $MWU_ITERS -learning_rate $LR -noise_type $NOISE_TYPE -alpha $ALPHA -opt_iters $OPT_ITERS"
eval $CMD
