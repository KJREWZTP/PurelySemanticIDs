#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

DOMAIN="toys"

BASE_EMBED_MODEL="bert-base-uncased"

BASE="t5-base"

RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"


LOG_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/logs/$DOMAIN"

SEMANTIC_ID_MODEL_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/$BASE"

# Parameter ranges
CODEBOOK_SIZES=(256)
LEARNING_RATES=(8e-4)
BATCH_SIZES=(32)

# History item text as input, predict target item id
for CODEBOOK_SIZE in "${CODEBOOK_SIZES[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      echo "Running experiment for CODEBOOK_SIZE=$CODEBOOK_SIZE, LEARNING_RATE=$LR, BATCH_SIZE=$BATCH_SIZE"

      NUM_CODES=3
      DEPTH=2
      ID_MODE="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}"
      CKPT_NAME="$ID_MODE"

      DATA_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data/$DOMAIN/query_retrieval/$ID_MODE/$BASE_EMBED_MODEL"
      CHECKPOINT_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/ckpt/${DOMAIN}_product/$CKPT_NAME"

      torchrun --nproc_per_node=2 --master_port=40016 $DOWNSTREAM_DIR/src/GEU/run_ours.py \
        --model_name_or_path "$SEMANTIC_ID_MODEL_DIR" \
        --do_train \
        --do_eval \
        --do_predict \
        --save_steps 1500 \
        --eval_steps 15000\
        --logging_steps 3000 \
        --max_steps 15000 \
        --learning_rate "$LR" \
        --max_source_length 1024 \
        --max_target_length 128 \
        --add_code_as_special_token True \
        --num_codes "$NUM_CODES" \
        --codebook_size "$CODEBOOK_SIZE" \
        --train_file "$DATA_DIR/train.json" \
        --validation_file "$DATA_DIR/val.json" \
        --test_file "$DATA_DIR/test.json" \
        --all_id_txt "$DATA_DIR/ids.txt" \
        --num_beams 20 \
        --output_dir "$CHECKPOINT_DIR/$BASE/LR_${LR}_BS_${BATCH_SIZE}_CB_${CODEBOOK_SIZE}" \
        --logging_dir "$LOG_DIR/$BASE/LR_${LR}_BS_${BATCH_SIZE}_CB_${CODEBOOK_SIZE}" \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --overwrite_output_dir \
        --predict_with_generate \
        --report_to none \
        --evaluation_strategy steps \
        --dataloader_num_workers 8 \
        --include_inputs_for_metrics True \
        --project_name 'query retrieval '$DOMAIN \
        --task 'retrieval'
    done
  done
done
