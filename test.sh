#!/usr/bin/env bash

DATE="$(date +%m-%d)"

EXP_DIR="test-$DATE"
ALPHA=.5
NUM_CLASSIFIERS=5
NUM_POINTS=10
NOISE_FUNC="gradientDescent"
ITERS=10
LOG_FILE="test.log"

CMD="python -m binary_experiments -classes 4 9 -noise_func $NOISE_FUNC -iters $ITERS -num_classifiers $NUM_CLASSIFIERS -exp_dir $EXP_DIR -alpha $ALPHA -num_point $NUM_POINTS -log_file $LOG_FILE"
eval $CMD