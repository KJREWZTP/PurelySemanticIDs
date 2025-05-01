#!/bin/bash

OUTPUT_DIR="retrieval-data/NQ/rqvae-256-depth3/bert-base-uncased"

NOCONFLICT_OUTPUT_DIR="retrieval-data/NQ/rqvae-256-depth3-noconflict-candidate/bert-base-uncased"

TEST_PATH="$OUTPUT_DIR/test.json"
ID_PATH="$OUTPUT_DIR/ids.txt"

NOCONFLICT_TEST_PATH="$NOCONFLICT_OUTPUT_DIR/test.json"
NOCONFLICT_ID_PATH="$NOCONFLICT_OUTPUT_DIR/ids.txt"

python split_test_w_conflict.py --test $TEST_PATH --ids $ID_PATH --output $OUTPUT_DIR --noconflict_test $NOCONFLICT_TEST_PATH --noconflict_output $NOCONFLICT_OUTPUT_DIR
