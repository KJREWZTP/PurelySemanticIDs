TOKENIZER='bert-base-uncased'
DOMAIN=beauty

INPUT_DIR=../../data/rec-data/$DOMAIN/preprocess
OUTPUT_DIR=../../data/rec-data/$DOMAIN/preprocess

python fast_tokenize.py \
        --input_dir $INPUT_DIR \
        --output $OUTPUT_DIR \
        --tokenizer $TOKENIZER
