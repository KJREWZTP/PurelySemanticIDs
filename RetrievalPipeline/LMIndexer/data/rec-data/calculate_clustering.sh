DOMAIN=beauty
DATA_DIR=./$DOMAIN/preprocess

BASE_MODEL=bert-base-uncased
MODE=ours

python calculate_clustering.py \
    --data_dir $DATA_DIR \
    --base $BASE_MODEL \
    --semantic_id_mode $MODE
