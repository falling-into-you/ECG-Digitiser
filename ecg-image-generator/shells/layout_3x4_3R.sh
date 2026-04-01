source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecg_image_gen
python gen_ecg_images_from_data_batch.py \
    -i data/s40689238 \
    -o outputs/layout_tests/3x4_3R \
    --config_file config_3x4.yaml \
    --num_columns 4 \
    --full_mode "V1,II,V5" \
    --max_num_images 1 \
    --image_only