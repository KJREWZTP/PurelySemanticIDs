#!/bin/bash

OUTPUT_DIR="retrieval-data/NQ/rqvae-256-depth3/bert-base-uncased"

TRAIN_PATH="$OUTPUT_DIR/train.json"
TEST_PATH="$OUTPUT_DIR/test.json"

NOCONFLICT_OUTPUT_DIR="retrieval-data/NQ/rqvae-256-depth3-noconflict-candidate/bert-base-uncased"

NOCONFLICT_TEST_PATH="$NOCONFLICT_OUTPUT_DIR/test.json"
THRESHOLD=1

python cold_start_test_gen.py --train $TRAIN_PATH --test $TEST_PATH --output $OUTPUT_DIR --threshold $THRESHOLD --noconflict_test $NOCONFLICT_TEST_PATH --noconflict_output $NOCONFLICT_OUTPUT_DIR
