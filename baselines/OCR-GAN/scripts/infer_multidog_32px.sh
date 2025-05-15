#!/usr/bin/env bash
set -e
exec > output_infer_multidog_32px.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever" "labrador_retriever" "teddy")
# DATASETS=("bottle")

INPUT_FILE_NAME="test_input_files_multidog_32px.txt"
MODEL_DATA_DIR="./data_multidog_32px"

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Infering on dataset: $DATASET_NAME"

OUTPUT_FILE_NAME="an_scores_${DATASET_NAME}_32px.csv"
DATASET_DIR="$(pwd)/../../data/class-based/example"

rm -rf "$MODEL_DATA_DIR"
ln -sfn "$DATASET_DIR" "$MODEL_DATA_DIR"

python csv_to_txt.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=3 python infer.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --manualseed 1 \
    --load_weights \
    --in_file "$INPUT_FILE_NAME" \
    --outf "$(pwd)/output_multidog_32px" \
    --out_file "$OUTPUT_FILE_NAME"
    # --save_image_freq 5
    # --save_test_images \
    # --workers 8 \
    # --nc 3 \
    # --device gpu \
    # --beta1 0.5 \
    # --w_adv 1 \
    # --w_con 50 \
    # --w_lat 1

python merge_csv.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
