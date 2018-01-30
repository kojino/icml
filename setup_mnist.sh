#!/bin/bash
#
#SBATCH -t 0-01:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=5000
#SBATCH --constraint=cuda-7.5
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

CMD="python -m setup_mnist"
eval $CMD
