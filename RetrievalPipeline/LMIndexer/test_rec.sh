# This is the code for sequential recommendation testing.
#!/bin/bash

base=t5-base
LR=5e-4

DOMAIN=toys
BASE_EMBED_MODEL=bert-base-uncased                # all-mpnet-base-v2
RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"


NUM_CODES=3
CODEBOOK_SIZE=256
DEPTH=3
BATCH_SIZE=32
LR=5e-4

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"

# ID_MODE="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}-noconflict-candidate"
ID_MODE="hcindexer-256,96-depth2-noconflict-recursive"

DATA_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data/$DOMAIN/sequential_retrieval/$ID_MODE/$BASE_EMBED_MODEL"
CHECKPOINT_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/ckpt/$DOMAIN/$ID_MODE"

LOAD_CHECKPOINT="$CHECKPOINT_DIR/$base/LR_${LR}_BS_${BATCH_SIZE}_CB_${CODEBOOK_SIZE}/checkpoint-15000"

LOG_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/logs/$DOMAIN"

export CUDA_VISIBLE_DEVICES=5

torchrun --nproc_per_node=1 --master_port 19296 $DOWNSTREAM_DIR/src/GEU/run_ours.py \
    --model_name_or_path $LOAD_CHECKPOINT \
    --do_predict \
    --max_source_length 1024 \
    --max_target_length 32 \
    --add_code_as_special_token True \
    --max_steps 0 \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --report_to none \
    --output_dir $LOAD_CHECKPOINT/result  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --task 'recommendation' \
    --eval_topk 5