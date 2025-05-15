#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate skipganomaly

DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/Industry/visa/$DATASET_NAME"

rm -rf './data_visa/'
mkdir -p "./data_visa/train/"
mkdir -p "./data_visa/test/"

ln -sn "$DATASET_DIR/train/good" "./data_visa/train/0.normal"

if [ -d "$DATASET_DIR/test/good" ]; then
  ln -sn "$DATASET_DIR/test/good" "./data_visa/test/0.normal"
else
  ln -sn "$DATASET_DIR/test/good_level_0" "./data_visa/test/0.normal"
fi

ln -sn "$DATASET_DIR/test/bad" "./data_visa/test/1.abnormal"

CUDA_VISIBLE_DEVICES=7 python train.py \
    --model skipganomaly \
    --dataset $DATASET_NAME \
    --dataroot "./data_visa" \
    --isize 32 \
    --niter 50 \
    --outf "./output_visa/" \
    --manualseed 1
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
