#!/bin/bash
#
#SBATCH -t 3-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

NOISE_TYPE="untargeted"
ALPHA=.5
EXP_TYPE="multiclass"
MWU_ITERS=20
OPT_ITERS=3000
LR=.001
DATA_PATH="multiclass_data_2"

CMD="python -m mnist_dl_experiments -data_path $DATA_PATH -mwu_iters $MWU_ITERS -learning_rate $LR -noise_type $NOISE_TYPE -alpha $ALPHA -opt_iters $OPT_ITERS"
eval $CMD
