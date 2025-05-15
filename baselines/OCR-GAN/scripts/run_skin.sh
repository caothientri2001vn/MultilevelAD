#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

INPUT_FILE_NAME="test_input_files_skin.txt"
MODEL_DATA_DIR="./data_skin"
MODEL_OUTPUT_DIR="./output_skin/"
DATASETS=("skin")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/Medical/skin-lesion"
OUTPUT_FILE_NAME="result_csvs/ocr-gan_skin-lesion.csv"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train"
mkdir -p "$MODEL_DATA_DIR/test/good"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/real_test"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/good/$(basename "$file")"
done

find "$DATASET_DIR/NV_level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/bad/$(basename "$file")"
done

python csv_to_txt.py "$(pwd)/../../data/template/skin-lesion_template.csv" "$INPUT_FILE_NAME"

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

python merge_csv.py "$(pwd)/../../data/template/skin-lesion_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
