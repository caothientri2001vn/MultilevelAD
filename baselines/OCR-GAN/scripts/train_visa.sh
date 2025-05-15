#!/usr/bin/env bash
set -e
exec > output_visa.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

MODEL_DATA_DIR="./data_visa"
MODEL_OUTPUT_DIR="./output_visa/"
DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")
# DATASETS=("hazelnut")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train/"
mkdir -p "$MODEL_DATA_DIR/test/"

ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/train/good"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/test/good"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/test/bad"

CUDA_VISIBLE_DEVICES=2 python train_all.py \
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
