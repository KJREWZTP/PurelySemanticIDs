#!/bin/bash
export PYTHONUNBUFFERED=1

BASE_MODEL="bert-base-uncased"
RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"

DOMAIN="sports"
EMBEDDING_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data/$DOMAIN/preprocess"
CODE_SAVE_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data/$DOMAIN/preprocess"

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"


CLUSTER_NUMS="256,96"
MAX_DEPTH=2           
MAX_DOCS=128

ID_MODE="hcindexer-${CLUSTER_NUMS}-depth${MAX_DEPTH}"

echo "Running HCIndexer clustering for DOMAIN=$DOMAIN"


# Run HCIndexer clustering
python "$DOWNSTREAM_DIR/src/hcindexer/hcindexer.py" \
    --embedding_dir "$EMBEDDING_DIR/$BASE_MODEL-embed.pt" \
    --output_path "$CODE_SAVE_DIR/" \
    --semantic_id "$ID_MODE" \
    --base_model "$BASE_MODEL" \
    --cluster_nums "$CLUSTER_NUMS" \
    --max_depth "$MAX_DEPTH" \
    --max_docs "$MAX_DOCS"

# Retrieval step
echo "Running retrieval generation"

RETRIEVAL_DATA_PATH="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data"
SAVE_DIR="$RETRIEVAL_DATA_PATH/$DOMAIN"

python "$RETRIEVAL_DATA_PATH/sqrec_gen.py" \
    --data_dir "$SAVE_DIR" \
    --train_iter 5 \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$ID_MODE"

SEMANTIC_ID_1="$ID_MODE-noconflict-candidate"

python "$RETRIEVAL_DATA_PATH/sqrec_gen.py" \
    --data_dir "$SAVE_DIR" \
    --train_iter 5 \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$SEMANTIC_ID_1"

SEMANTIC_ID_2="$ID_MODE-noconflict-recursive"

python "$RETRIEVAL_DATA_PATH/sqrec_gen.py" \
    --data_dir "$SAVE_DIR" \
    --train_iter 5 \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$SEMANTIC_ID_2"

echo "Retrieval generation complete"

RAW_RETRIEVAL_DATA_DIR=$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/esci-data/shopping_queries_dataset
RAW_ALL_ITEM_FILE=$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/metadata_2014/metadata.json
INTERMEDIATE_DIR=$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data
RETRIEVAL_DATA_PATH="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/rec-data"

echo "Running product search generation for CODEBOOK_SIZE=$CODEBOOK_SIZE, DEPTH=$DEPTH"

python "$RETRIEVAL_DATA_PATH/pdretr_gen.py" \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --domain $DOMAIN \
    --base $BASE_MODEL \
    --semantic_id_mode "$ID_MODE"

python "$RETRIEVAL_DATA_PATH/pdretr_gen.py" \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --domain $DOMAIN \
    --base $BASE_MODEL \
    --semantic_id_mode "$SEMANTIC_ID_1"

python "$RETRIEVAL_DATA_PATH/pdretr_gen.py" \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --domain $DOMAIN \
    --base $BASE_MODEL \
    --semantic_id_mode "$SEMANTIC_ID_2"