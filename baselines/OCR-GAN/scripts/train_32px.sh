#!/usr/bin/env bash
set -e
exec > output_32px.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable")
# DATASETS=("bottle")

INPUT_FILE_NAME="test_input_files_32px.txt"
MODEL_DATA_DIR="./data_32px"

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/mvtec_order/$DATASET_NAME"
OUTPUT_FILE_NAME="csv_rerun/ocr-gan-32px_mvtec_${DATASET_NAME}.csv"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train/"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/train/good"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/test/good"
ln -sn "$DATASET_DIR/test" "$MODEL_DATA_DIR/real_test"

# Process all other subdirectories of test
for subdir in "$DATASET_DIR/test"/*; do
  if [ -d "$subdir" ] && [ "$(basename "$subdir")" != "good_level_0" ]; then
    subdir_name=$(basename "$subdir")

    # Process each file in the subdirectory
    for file in "$subdir"/*; do
      if [ -f "$file" ]; then
        ln -sfn "$file" "$MODEL_DATA_DIR/test/bad/${subdir_name}_$(basename "$file")"
      fi
    done
  fi
done

cut -d',' -f1 "$DATASET_DIR/${DATASET_NAME}_template.csv" | tail -n +2 > "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=3 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --outf "./output_32px" \
    --niter 50 \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
