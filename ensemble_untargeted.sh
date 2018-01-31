#!/bin/bash
#
#SBATCH -t 3-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

NOISE_TYPE="untargeted"

CMD="python -m imagenet_ensemble -noise_type $NOISE_TYPE"
eval $CMD
