#!/bin/bash
#SBATCH -t 1-0:00
#SBATCH -p shared
#SBATCH --mem=1000
#SBATCH -o sample_linear_experiment.out
#SBATCH -e sample_linear_experiment.err
#SBATCH --mail-type=all
#SBATCH --mail-user=kojinoshiba@college.harvard.edu

ALPHA=.5
EXP_TYPE="binary"
NUM_CLASSIFIERS=100
NUM_POINTS=100
NOISE_FUNC="greedyAscent"
ITERS=100
DATA_PATH="binary_data_2"
LIST_NUM_SAMPLES=$(seq 100)

# Full set of classifiers
SAMPLE="none"
CMD="python -m linear_experiments -data_path $DATA_PATH -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_type $EXP_TYPE -alpha $ALPHA -sample $SAMPLE"
./sample_linear_experiment_helper.sh $CMD

# Sample classifiers once
SAMPLE="once"
for NUM_SAMPLES in $LIST_NUM_SAMPLES
do
  CMD="python -m linear_experiments -data_path $DATA_PATH -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_type $EXP_TYPE -alpha $ALPHA -sample $SAMPLE -num_samples $NUM_SAMPLES"
  ./sample_linear_experiment_helper.sh $CMD
done

# Sample classifiers every iteration
SAMPLE="iter"
for NUM_SAMPLES in $LIST_NUM_SAMPLES
do
  CMD="python -m linear_experiments -data_path $DATA_PATH -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_type $EXP_TYPE -alpha $ALPHA -sample $SAMPLE -num_samples $NUM_SAMPLES"
  ./sample_linear_experiment_helper.sh $CMD
done
