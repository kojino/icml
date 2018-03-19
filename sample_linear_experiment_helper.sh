#!/bin/bash
#SBATCH -t 1-0:00
#SBATCH -p shared
#SBATCH --mem=5000
#SBATCH -o run_sample_linear_experiment_%j.out
#SBATCH -e run_sample_linear_experiment_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=kojinoshiba@college.harvard.edu

eval $1
