#!/bin/bash
#
#SBATCH -t 7-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

NOISE_TYPE="untargeted"
ALPHA=2.2
DATA_SET="mnist"
MWU_ITERS=2
OPT_ITERS=3000
LR=.001

CMD="python -m deep_learning_experiments -data_set $DATA_SET -mwu_iters $MWU_ITERS -learning_rate $LR -noise_type $NOISE_TYPE -alpha $ALPHA -opt_iters $OPT_ITERS"
eval $CMD
