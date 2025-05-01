domain=sports # sports, toys

mkdir $domain/

python data_preprocess.py \
    --data_dir ../raw-data \
    --output_dir ./$domain \
    --short_data_name $domain
