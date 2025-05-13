#!/usr/bin/env bash
set -e
exec > output_infer.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

# DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")
DATASETS=("bottle")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Infering on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/mvtec/$DATASET_NAME"
MODEL_DATA_DIR='./data'

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train/"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/train/good"
ln -sn "$DATASET_DIR/test/good" "$MODEL_DATA_DIR/test/good"

# Process all other subdirectories of test
for subdir in "$DATASET_DIR/test"/*; do
  if [ -d "$subdir" ] && [ "$(basename "$subdir")" != "good" ]; then
    subdir_name=$(basename "$subdir")

    # Process each file in the subdirectory
    for file in "$subdir"/*; do
      if [ -f "$file" ]; then
        ln -sn "$file" "$MODEL_DATA_DIR/test/bad/${subdir_name}_$(basename "$file")"
      fi
    done
  fi
done

CUDA_VISIBLE_DEVICES=3 python test.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --manualseed 1 \
    --load_weights \
    --outf "./output_32px"

done
