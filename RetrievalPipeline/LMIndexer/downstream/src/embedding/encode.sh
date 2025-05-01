BASE_MODEL='bert-base-uncased' # bert-base-uncased

DOMAIN=toys
DATA_DIR=../../data/rec-data/$DOMAIN/preprocess
OUTPUT_DIR=../../data/rec-data/$DOMAIN/preprocess

python encode.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --tokenizer $BASE_MODEL \
        --model_name $BASE_MODEL \
        --batch_size 512 \
        --max_length 256 \
        --dataloader_num_workers 0
