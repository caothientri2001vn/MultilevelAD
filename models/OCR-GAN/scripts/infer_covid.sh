#!/usr/bin/env bash
set -e
exec > output_infer_covid.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

# DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever" "labrador_retriever" "teddy")
DATASETS=("covid")

INPUT_FILE_NAME="test_input_files_covid.txt"
MODEL_DATA_DIR="./data_covid"

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Infering on dataset: $DATASET_NAME"

OUTPUT_FILE_NAME="an_scores_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/severity-based/covid19"

rm -rf "$MODEL_DATA_DIR"
ln -sfn "$DATASET_DIR" "$MODEL_DATA_DIR"

python csv_to_txt.py "$DATASET_DIR/covid19_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=3 python infer.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 256 \
    --manualseed 1 \
    --load_weights \
    --in_file "$INPUT_FILE_NAME" \
    --outf "$(pwd)/output_covid" \
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

python merge_csv.py "$DATASET_DIR/covid19_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
