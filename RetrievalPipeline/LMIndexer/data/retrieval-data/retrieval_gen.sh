DOMAIN=NQ # NQ_aug

DATA_DIR=../raw-data/$DOMAIN
SAVE_DIR=./$DOMAIN

SEMANTIC_MODE=rqvae-48,48,48
BASE_MODEL='bert-base-uncased'

python retrieval_gen.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --base $BASE_MODEL \
    --semantic_id_mode $SEMANTIC_MODE
