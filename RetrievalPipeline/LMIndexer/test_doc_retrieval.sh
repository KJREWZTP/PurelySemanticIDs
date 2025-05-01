# This is the code for document retrieval testing.
#!/bin/bash
base=t5-base

DOMAIN="NQ"
BASE_EMBED_MODEL=bert-base-uncased
RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"


NUM_CODES=4
CODEBOOK_SIZE=256
DEPTH=3
BATCH_SIZE=1024
LR=2e-3

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"


ID_MODE="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}"

DATA_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/data/retrieval-data/$DOMAIN/$ID_MODE/$BASE_EMBED_MODEL"
CHECKPOINT_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/ckpt/$DOMAIN/$ID_MODE"

LOAD_CHECKPOINT="$CHECKPOINT_DIR/$base/LR_${LR}_BS_${BATCH_SIZE}_CB_${CODEBOOK_SIZE}/checkpoint-30000"

LOG_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/logs/$DOMAIN"

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port 19289 $DOWNSTREAM_DIR/src/GEU/run_ours.py \
    --model_name_or_path $LOAD_CHECKPOINT \
    --do_predict \
    --max_source_length 1024 \
    --max_target_length 128 \
    --add_code_as_special_token True \
    --max_steps 0 \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/unique_test_cases.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --report_to none \
    --output_dir $LOAD_CHECKPOINT/result  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --predict_with_generate \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --task 'retrieval_levels' \
    --eval_topk 10