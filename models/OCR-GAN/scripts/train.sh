#!/usr/bin/env bash
set -e
exec > output.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

# DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")
DATASETS=("zipper" "bottle")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/mvtec/$DATASET_NAME"

rm -rf './data/'
mkdir -p "./data/train/"
mkdir -p "./data/test/bad"

ln -sfn "$DATASET_DIR/train/good" "./data/train/good"
ln -sfn "$DATASET_DIR/test/good" "./data/test/good"

# Process all other subdirectories of test
for subdir in "$DATASET_DIR/test"/*; do
  if [ -d "$subdir" ] && [ "$(basename "$subdir")" != "good" ]; then
    subdir_name=$(basename "$subdir")

    # Process each file in the subdirectory
    for file in "$subdir"/*; do
      if [ -f "$file" ]; then
        ln -sfn "$file" "./data/test/bad/${subdir_name}_$(basename "$file")"
      fi
    done
  fi
done

CUDA_VISIBLE_DEVICES=6 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "./data" \
    --isize 256 \
    --niter 70
    # --save_image_freq 5
    # --save_test_images \
    # --workers 8 \
    # --nc 3 \
    # --device gpu \
    # --outf "$(pwd)/output" \
    # --beta1 0.5 \
    # --w_adv 1 \
    # --w_con 50 \
    # --w_lat 1
done
