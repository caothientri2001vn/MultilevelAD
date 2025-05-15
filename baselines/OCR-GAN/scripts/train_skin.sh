#!/usr/bin/env bash
set -e
exec > output_skin.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

MODEL_DATA_DIR="./data_skin"
MODEL_OUTPUT_DIR="./output_skin/"
DATASETS=("skin")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/severity-based/skin-lesion"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train"
mkdir -p "$MODEL_DATA_DIR/test/good"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sfn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/test/good/$(basename "$file")"
done

find "$DATASET_DIR/NV_level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/test/bad/$(basename "$file")"
done

CUDA_VISIBLE_DEVICES=3 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 256 \
    --niter 70 \
    --outf "$MODEL_OUTPUT_DIR"
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
