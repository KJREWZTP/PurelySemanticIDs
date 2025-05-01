#!/bin/bash

BASE_MODEL="bert-base-uncased"
RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"

DOMAIN="macro"
EMBEDDING_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"
CODE_SAVE_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"

export CUDA_VISIBLE_DEVICES=1


# Parameter ranges
CODEBOOK_SIZES=(1024)
CODE_EMBEDDING_DIMS=(32)
DEPTHS=(3)
LEARNING_RATES=(0.001)
BATCH_SIZES=(1024)

# Number of epochs and workers
NUM_EPOCHS=150
NUM_WORKERS=32

for CODEBOOK_SIZE in "${CODEBOOK_SIZES[@]}"; do
  for CODE_EMBEDDING_DIM in "${CODE_EMBEDDING_DIMS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
      for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
          echo "Running training for CODEBOOK_SIZE=$CODEBOOK_SIZE, CODE_EMBEDDING_DIM=$CODE_EMBEDDING_DIM, DEPTH=$DEPTH, LEARNING_RATE=$LEARNING_RATE, BATCH_SIZE=$BATCH_SIZE"

          # Training command
          python "$DOWNSTREAM_DIR/src/rqvae/vec_rqvae.py" \
            --embedding_dir "$EMBEDDING_DIR" \
            --code_save_dir "$CODE_SAVE_DIR" \
            --base_model "$BASE_MODEL" \
            --code_embedding_dim "$CODE_EMBEDDING_DIM" \
            --codebook_size "$CODEBOOK_SIZE" \
            --depth "$DEPTH" \
            --num_epochs "$NUM_EPOCHS" \
            --learning_rate "$LEARNING_RATE" \
            --batch_size "$BATCH_SIZE" \
            --domain "$DOMAIN" \
            --num_workers "$NUM_WORKERS"

          echo "Running inference for CODEBOOK_SIZE=$CODEBOOK_SIZE, CODE_EMBEDDING_DIM=$CODE_EMBEDDING_DIM, DEPTH=$DEPTH, LEARNING_RATE=$LEARNING_RATE, BATCH_SIZE=$BATCH_SIZE"

          SEMANTIC_MODE="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}"

          Inference command
          python "$DOWNSTREAM_DIR/src/rqvae/vec_rqvae_infer.py" \
            --embedding_dir "$EMBEDDING_DIR" \
            --code_save_dir "$CODE_SAVE_DIR" \
            --base_model "$BASE_MODEL" \
            --code_embedding_dim "$CODE_EMBEDDING_DIM" \
            --codebook_size "$CODEBOOK_SIZE" \
            --depth "$DEPTH" \
            --num_epochs "$NUM_EPOCHS" \
            --learning_rate "$LEARNING_RATE" \
            --batch_size "$BATCH_SIZE" \
            --domain "$DOMAIN" \
            --num_workers "$NUM_WORKERS" \
            --semantic_id_mode "$SEMANTIC_MODE" \

          DATA_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"
          RETRIEVAL_DATA_PATH="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/retrieval-data"
          SAVE_DIR="$RETRIEVAL_DATA_PATH/$DOMAIN"

          echo "Running retrieval generation for CODEBOOK_SIZE=$CODEBOOK_SIZE, DEPTH=$DEPTH"

          python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
              --data_dir "$DATA_DIR" \
              --save_dir "$SAVE_DIR" \
              --base "$BASE_MODEL" \
              --semantic_id_mode "$SEMANTIC_MODE"

          SEMANTIC_MODE_NOCONFLICT_1="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}-noconflict-candidate"
          
          python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
              --data_dir "$DATA_DIR" \
              --save_dir "$SAVE_DIR" \
              --base "$BASE_MODEL" \
              --semantic_id_mode "$SEMANTIC_MODE_NOCONFLICT_1"
          
          SEMANTIC_MODE_NOCONFLICT_2="rqvae-${CODEBOOK_SIZE}-depth${DEPTH}-noconflict-recursive"
          
          python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
              --data_dir "$DATA_DIR" \
              --save_dir "$SAVE_DIR" \
              --base "$BASE_MODEL" \
              --semantic_id_mode "$SEMANTIC_MODE_NOCONFLICT_2"
        done
      done
    done
  done
done
