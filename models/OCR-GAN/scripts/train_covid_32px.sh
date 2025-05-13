#!/usr/bin/env bash
set -e
exec > output_covid_32px.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

INPUT_FILE_NAME="test_input_files_covid_32px.txt"
MODEL_DATA_DIR="./data_covid_32px"
MODEL_OUTPUT_DIR="./output_covid_32px/"
DATASETS=("covid")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/severity-based/covid19"
OUTPUT_FILE_NAME="csv_rerun/ocr-gan-32px_${DATASET_NAME}.csv"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train"
mkdir -p "$MODEL_DATA_DIR/test/good"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/real_test"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1.0" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/bad/$(basename "$file")"
done

python csv_to_txt.py "$DATASET_DIR/${DATASET_NAME}19_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=3 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --niter 50 \
    --outf "$MODEL_OUTPUT_DIR" \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$DATASET_DIR/${DATASET_NAME}19_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
