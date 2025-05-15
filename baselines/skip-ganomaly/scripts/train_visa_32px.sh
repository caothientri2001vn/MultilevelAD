#!/usr/bin/env bash
set -e
exec > output_visa_32px.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate skipganomaly

DATASETS=("candle_not" "capsules" "cashew_not" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")
# DATASETS=("hazelnut")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"

rm -rf './data_32px/'
mkdir -p "./data_32px/train/"
mkdir -p "./data_32px/test/"

ln -sfn "$DATASET_DIR/train/good" "./data_32px/train/0.normal"

if [ -d "$DATASET_DIR/test/good" ]; then
  ln -sfn "$DATASET_DIR/test/good" "./data_32px/test/0.normal"
else
  ln -sfn "$DATASET_DIR/test/good_level_0" "./data_32px/test/0.normal"
fi

ln -sfn "$DATASET_DIR/test/bad" "./data_32px/test/1.abnormal"

CUDA_VISIBLE_DEVICES=7 python train.py \
    --model skipganomaly \
    --dataset $DATASET_NAME \
    --dataroot "./data_32px" \
    --isize 32 \
    --niter 50 \
    --outf "./output_visa_32px/" \
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
