#!/usr/bin/env bash
set -e
exec > logs/infer_skin_$1.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

DATASETS=("skin")

INPUT_FILE_NAME="input_list/skin_${1}.txt"
MODEL_DATA_DIR="./dataset_skin_${1}"
MODEL_OUTPUT_DIR="./output_32px_skin_$1"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="csv_corrupted/ocr-gan-32px_skin-lesion_corrupted_$1.csv"
DATASET_DIR="$(pwd)/../../data/severity-based/skin-lesion"
TEST_DATASET_DIR="$(pwd)/../../data/corrupted/$1/skin-lesion"

if [[ -d $MODEL_DATA_DIR ]]; then
  rm -r "$MODEL_DATA_DIR"
fi

mkdir -p "$MODEL_DATA_DIR/train"
mkdir -p "$MODEL_DATA_DIR/test/good"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/real_test"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/good/$(basename "$file")"
done

find "$DATASET_DIR/NV_level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/bad/$(basename "$file")"
done

python csv_to_txt.py "$TEST_DATASET_DIR/skin-lesion_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=$2 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --niter 50 \
    --outf "$MODEL_OUTPUT_DIR" \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$TEST_DATASET_DIR/skin-lesion_template.csv" "$OUTPUT_FILE_NAME" "merged_${OUTPUT_FILE_NAME}"
done
