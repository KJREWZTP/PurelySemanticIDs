#!/bin/bash
export PYTHONUNBUFFERED=1

BASE_MODEL="bert-base-uncased"
RETRIEVAL_PIPELINE_DIR_PATH="../RetrievalPipeline"

DOMAIN="NQ"
EMBEDDING_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"
CODE_SAVE_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"

DOWNSTREAM_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/downstream"

# Parameter ranges
CLUSTER_NUMS="512, 128"  # Cluster sizes per depth level
MAX_DEPTH=2            # Maximum depth for recursive clustering
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

DATA_DIR="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/raw-data/$DOMAIN"
RETRIEVAL_DATA_PATH="$RETRIEVAL_PIPELINE_DIR_PATH/LMIndexer/data/retrieval-data"
SAVE_DIR="$RETRIEVAL_DATA_PATH/$DOMAIN"
          
python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$SEMANTIC_MODE"

SEMANTIC_ID_1="$ID_MODE-noconflict-candidate"

python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$SEMANTIC_ID_1"

SEMANTIC_ID_2="$ID_MODE-noconflict-recursive"

python "$RETRIEVAL_DATA_PATH/retrieval_gen.py" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --base "$BASE_MODEL" \
    --semantic_id_mode "$SEMANTIC_ID_2"

echo "Retrieval generation complete"