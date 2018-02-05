#!/bin/bash
#
#SBATCH -t 2-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

NOISE_TYPE="untargeted"
OPT_ITERS=3000
ALPHA=300
LR=.01
MODEL=0

CMD="python -m baselines_imagenet -noise_type $NOISE_TYPE -opt_iters $OPT_ITERS -alpha $ALPHA -model $MODEL -learning_rate $LR"
eval $CMD
